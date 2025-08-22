#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, time, cv2, numpy as np
import argparse
from tqdm import tqdm

import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
import math

# ---- 原有正脸模板（112x112） ----
_ARCFACE_5PTS_112 = np.array([
    [38.2946, 51.6963],  # left_eye
    [73.5318, 51.5014],  # right_eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left_mouth
    [70.7299, 92.2041],  # right_mouth
], dtype=np.float32)

def _scale(pts, size):
    s = size / 112.0
    return (pts * s).astype(np.float32)

def _yaw_templates(size=512):
    base = _ARCFACE_5PTS_112.copy()
    def tweak(xshift_eye=0.0, xshift_nose=0.0, xshift_mouth=0.0):
        t = base.copy()
        t[[0,3],0] -= xshift_eye
        t[[1,4],0] += xshift_eye
        t[2,0]     += xshift_nose
        t[[3,4],0] += xshift_mouth
        return _scale(t, size)
    return [
        _scale(base, size),
        tweak(xshift_eye=1.5,  xshift_nose=1.0,  xshift_mouth=0.7),
        tweak(xshift_eye=-1.5, xshift_nose=-1.0, xshift_mouth=-0.7),
        tweak(xshift_eye=3.0,  xshift_nose=2.2,  xshift_mouth=1.6),
        tweak(xshift_eye=-3.0, xshift_nose=-2.2, xshift_mouth=-1.6)
    ]

def _affine_and_error(src5, dst5, img_bgr, size):
    M, inliers = cv2.estimateAffinePartial2D(src5, dst5, method=cv2.LMEDS)
    if M is None:
        return None, 1e9, None
    aligned = cv2.warpAffine(img_bgr, M, (size, size),
                             flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT)
    src_homo = np.hstack([src5, np.ones((5,1), dtype=np.float32)])
    proj = (src_homo @ M.T)
    err = float(np.sqrt(np.mean(np.sum((proj - dst5)**2, axis=1))))
    return aligned, err, M

def align_face_5pts_auto(img_bgr, kps5, size=512, err_thresh=3.5):
    best = (None, 1e9, None)
    for dst in _yaw_templates(size=size):
        aligned, err, M = _affine_and_error(kps5, dst, img_bgr, size)
        if err < best[1]:
            best = ( (aligned, M), err, dst )
    (pack, err, _dst) = best
    if pack is None or err > err_thresh:
        return None
    aligned, M = pack
    M_inv = cv2.invertAffineTransform(M)
    return aligned, M, M_inv

# --------- 预处理/后处理 ---------
def preprocess_bgr(img_bgr, size=512, aligner=None, margin_ratio=0.08):
    H, W = img_bgr.shape[:2]
    used_align, M_inv = False, None

    kps = None
    if aligner is not None:
        try:
            kps = aligner.get_5pts(img_bgr)
            if kps is not None:
                kps = kps.astype(np.float32)
        except Exception:
            kps = None

    aligned = None
    if kps is not None and kps.shape == (5,2):
        pack = align_face_5pts_auto(img_bgr, kps, size=size, err_thresh=3.5)
        if pack is not None:
            aligned, _M, M_inv = pack
            used_align = True

    if aligned is None:
        cx, cy = W/2.0, H/2.0
        side = int(round(max(W, H) * (1.0 - 0.00)))
        side = int(round(side * (1.0 + margin_ratio)))
        if side % 2: side += 1
        xs1 = max(0, int(round(cx - side/2)));  ys1 = max(0, int(round(cy - side/2)))
        xs2 = min(W, xs1 + side);               ys2 = min(H, ys1 + side)
        crop = img_bgr[ys1:ys2, xs1:xs2]
        ch, cw = crop.shape[:2]
        interp = cv2.INTER_AREA if max(ch, cw) > size else cv2.INTER_LANCZOS4
        scale = size / max(ch, cw)
        nh, nw = int(round(ch*scale)), int(round(cw*scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=interp)
        aligned = cv2.copyMakeBorder(
            resized, 0, size-nh, 0, size-nw, cv2.BORDER_REPLICATE
        )

    img = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)

    meta = {'orig_hw': (H, W), 'M_inv': M_inv, 'size': size, 'used_align': used_align}
    return np.ascontiguousarray(img), meta

def postprocess_chw(out_chw, target_hw):
    x = np.transpose(out_chw, (1, 2, 0))
    x = (x + 1.0) / 2.0
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255.0).astype(np.uint8)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    h, w = target_hw
    if x.shape[:2] != (h, w):
        x = cv2.resize(x, (w, h), interpolation=cv2.INTER_CUBIC)
    return x

def _enqueue_v3(context, stream):
    for name in ("enqueue_v3","enqueueV3","execute_async_v3","executeV3"):
        if hasattr(context, name):
            fn = getattr(context, name)
            try:    return fn(stream.handle)
            except TypeError: return fn(stream_handle=stream.handle)
    raise RuntimeError("No v3 execute method on context")

def _safe_imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(path, img)

def _hstack_compare(orig, enhanced, label_left="orig", label_right="enh"):
    h, w = orig.shape[:2]
    enhanced_rs = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LANCZOS4)
    band = max(28, h // 20)
    canvas = np.zeros((h + band, w * 2, 3), dtype=np.uint8)
    canvas[band:, :w] = orig
    canvas[band:, w:] = enhanced_rs
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, label_left,  (10, band - 8), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, label_right, (w + 10, band - 8), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return canvas

# --------- TensorRT Runner（支持 batch=8 静态） ---------
class TrtRunner:
    def __init__(self, engine_path: str, size: int = 512, device_id: int = 0):
        self.size = size
        self.ctx = None
        self.runtime = None
        self.engine = None
        self.context = None
        self.stream = None
        self.use_named = False
        self.in_name = None
        self.out_name = None
        self.in_dtype = None
        self.out_dtype = None
        self.in_shape = None
        self.out_shape = None
        self.batch_size = None
        self.d_in = None
        self.d_out = None

        cuda.init()
        self.dev = cuda.Device(device_id)
        self.ctx = self.dev.make_context()

        try:
            logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(logger)
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
            assert self.engine is not None
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()

            # I/O 描述
            self.use_named = not hasattr(self.engine, "get_binding_name") and hasattr(self.engine, "get_tensor_name")
            if self.use_named:
                names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
                ins  = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
                outs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
                assert len(ins)==1 and len(outs)==1
                self.in_name, self.out_name = ins[0], outs[0]
                shp_in  = tuple(self.engine.get_tensor_shape(self.in_name))
                shp_out = tuple(self.engine.get_tensor_shape(self.out_name))
                self.in_dtype  = trt.nptype(self.engine.get_tensor_dtype(self.in_name))
                self.out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.out_name))
            else:
                shp_in  = tuple(self.context.get_binding_shape(0))
                shp_out = tuple(self.context.get_binding_shape(1))
                self.in_name  = self.engine.get_binding_name(0)
                self.out_name = self.engine.get_binding_name(1)
                self.in_dtype  = trt.nptype(self.engine.get_binding_dtype(0))
                self.out_dtype = trt.nptype(self.engine.get_binding_dtype(1))

            # 静态 batch（例如 8x3x512x512）
            if -1 in shp_in:
                raise RuntimeError("当前引擎是动态 shape，本 Runner 逻辑假定为静态 batch。请先固定/设置输入形状。")

            self.in_shape  = tuple(int(x) for x in shp_in)    # e.g. (8,3,512,512)
            self.out_shape = tuple(int(x) for x in shp_out)   # e.g. (8,3,512,512)
            self.batch_size = self.in_shape[0]
            assert self.batch_size >= 1, "无效的 batch size"

            print(f"[INFO] Engine loaded: {engine_path}")
            print(f"[INFO] Input shape: {self.in_shape}, dtype={self.in_dtype}")
            print(f"[INFO] Output shape: {self.out_shape}, dtype={self.out_dtype}")
            print(f"[INFO] Batch size = {self.batch_size}")

            # 预分配显存
            self.d_in  = cuda.mem_alloc(int(np.prod(self.in_shape))  * np.dtype(self.in_dtype).itemsize)
            self.d_out = cuda.mem_alloc(int(np.prod(self.out_shape)) * np.dtype(self.out_dtype).itemsize)

            # 绑定
            if self.use_named:
                self.context.set_tensor_address(self.in_name,  int(self.d_in))
                self.context.set_tensor_address(self.out_name, int(self.d_out))
            else:
                self.bindings = [0]*self.engine.num_bindings
                self.bindings[0] = int(self.d_in)
                self.bindings[1] = int(self.d_out)
        except Exception:
            self.close()
            raise

    def _infer_batch(self, batch_tensor: np.ndarray):
        """batch_tensor: (B,3,H,W) -> 返回 (out_host, infer_sec)"""
        assert batch_tensor.shape == self.in_shape, f"expected {self.in_shape}, got {batch_tensor.shape}"
        t0 = time.time()
        # H2D
        cuda.memcpy_htod_async(self.d_in, batch_tensor, self.stream)
        # 执行
        if self.use_named:
            _enqueue_v3(self.context, self.stream)
        else:
            if hasattr(self.context, "execute_async_v2"):
                self.context.execute_async_v2(self.bindings, self.stream.handle)
            else:
                self.context.execute_v2(self.bindings)
        # D2H
        out_host = np.empty(self.out_shape, dtype=self.out_dtype)
        cuda.memcpy_dtoh_async(out_host, self.d_out, self.stream)
        self.stream.synchronize()
        infer_sec = time.time() - t0
        print(f"[INFO] Inference done, time={infer_sec:.4f}s, out shape={out_host.shape}")
        return out_host, infer_sec

    def run_batch(self, imgs_bgr):
        """
        imgs_bgr: List[np.ndarray]，长度应为 batch_size；若不足，会自动用最后一张补齐
        返回: (List[np.ndarray], infer_sec) —— infer_sec 为该批次纯推理时间
        """
        B = self.batch_size
        assert len(imgs_bgr) >= 1, "至少提供一张图像"
        valid = min(len(imgs_bgr), B)
        #print(f"[INFO] Running batch with {valid}/{B} images")
        # 预处理
        nets = []
        metas = []
        for i in range(valid):
            net_in, meta = preprocess_bgr(imgs_bgr[i], size=self.in_shape[-1])
            if isinstance(net_in, tuple):
                net_in = net_in[0]
            nets.append(net_in.astype(self.in_dtype, copy=False))
            metas.append(meta)
            #print(f"   └─ Preprocessed image[{i}] -> {net_in.shape}")
        # 不足 batch 的补齐（复用最后一张）
        while len(nets) < B:
            nets.append(nets[-1])
            metas.append(metas[-1])

        batch_tensor = np.concatenate(nets, axis=0)  # (B,3,H,W)
        out_host, infer_sec = self._infer_batch(batch_tensor)

        # 后处理（仅返回前 valid 个）
        outs = []
        for i in range(valid):
            H0, W0 = metas[i]['orig_hw'] if metas[i] is not None else (self.in_shape[-2], self.in_shape[-1])
            out_img = postprocess_chw(out_host[i].astype(np.float32, copy=False), (H0, W0))
            outs.append(out_img)
            #print(f"   └─ Postprocessed image[{i}] -> {out_img.shape}")
        return outs, infer_sec

    def run(self, img_bgr: np.ndarray):
        """兼容单图：内部自动补齐到 batch 运行，只返回第 1 张结果"""
        outs, _ = self.run_batch([img_bgr])
        return outs[0]

    def close(self):
        try:
            if self.stream is not None:
                self.stream.synchronize()
        except Exception:
            pass
        try:
            if self.d_in is not None:  self.d_in.free()
            if self.d_out is not None: self.d_out.free()
        except Exception:
            pass
        self.context = None
        self.engine = None
        self.runtime = None
        try:
            if self.ctx is not None:
                self.ctx.pop()
                self.ctx.detach()
        except Exception:
            pass
# --------- 批量处理并生成对比图（统计总耗时/平均耗时） ---------
def process_folder(engine_path, input_dir, output_dir, compare_dir, size=512):
    assert os.path.exists(engine_path), f"Engine not found: {engine_path}"
    assert os.path.isdir(input_dir), f"Input dir not found: {input_dir}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(exts)]
    if not files:
        print(f"输入目录中没有图片: {input_dir}")
        return

    runner = TrtRunner(engine_path, size=size)
    B = runner.batch_size

    total_images = 0
    total_batches = 0
    total_wall_sec = 0.0           # 端到端耗时（预处理+推理+后处理+IO）
    total_infer_sec = 0.0          # 仅推理耗时（H2D+执行+D2H）

    t_all_start = time.time()
    try:
        # 按 batch 分组
        for idx in tqdm(range(0, len(files), B), desc=f"GFPGAN增强(批={B})+对比图"):
            chunk = files[idx: idx+B]
            # print(f"\n[INFO] Processing batch idx={idx//B}, files={chunk}")

            # 读图
            imgs = []
            paths = []
            for fn in chunk:
                p = os.path.join(input_dir, fn)
                img = cv2.imread(p, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[WARN] 无法读取: {p}")
                    continue
                imgs.append(img); paths.append(p)
            if not imgs:
                continue

            # 端到端计时（本批）
            t0 = time.time()
            outs, infer_sec = runner.run_batch(imgs)  # 只返回有效张数的结果 + 该批纯推理耗时
            # 写盘
            for i, fn in enumerate(chunk[:len(outs)]):
                out_path = os.path.join(output_dir, fn)
                _safe_imwrite(out_path, outs[i])

                cmp_img = _hstack_compare(imgs[i], outs[i], label_left="original", label_right="enhanced")
                cmp_path = os.path.join(compare_dir, os.path.splitext(fn)[0] + "_compare.jpg")
                _safe_imwrite(cmp_path, cmp_img)
                # print(f"[SAVE] {fn} -> {out_path}, compare={cmp_path}")
            t1 = time.time()

            # 累计
            batch_wall = t1 - t0
            total_wall_sec += batch_wall
            total_infer_sec += infer_sec
            total_images += len(outs)
            total_batches += 1

            # print(f"[STAT] Batch#{total_batches}: wall={batch_wall:.4f}s, infer={infer_sec:.4f}s, imgs={len(outs)}")
    finally:
        runner.close()
    t_all_end = time.time()

    # ---- 汇总统计 ----
    grand_wall = t_all_end - t_all_start                 # 包含初始化与最后清理的整体耗时
    avg_wall_per_img_ms  = (total_wall_sec / max(total_images,1)) * 1000.0
    avg_infer_per_img_ms = (total_infer_sec / max(total_images,1)) * 1000.0
    avg_wall_per_batch_s = (total_wall_sec / max(total_batches,1))
    avg_infer_per_batch_s= (total_infer_sec / max(total_batches,1))

    print("\n================= SUMMARY =================")
    print(f"Total images processed : {total_images}")
    print(f"Total batches          : {total_batches} (batch_size={B})")
    print(f"Total wall time        : {total_wall_sec:.4f}s   (sum of per-batch end-to-end)")
    print(f"Total pure infer time  : {total_infer_sec:.4f}s   (sum of per-batch H2D+exec+D2H)")
    print(f"Grand wall time        : {grand_wall:.4f}s        (including init/finalize)")
    print(f"Avg wall per image     : {avg_wall_per_img_ms:.2f} ms/img")
    print(f"Avg infer per image    : {avg_infer_per_img_ms:.2f} ms/img")
    print(f"Avg wall per batch     : {avg_wall_per_batch_s:.4f} s/batch")
    print(f"Avg infer per batch    : {avg_infer_per_batch_s:.4f} s/batch")
    print("===========================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="/workspace/GFPGAN-onnxruntime-batch/GFPGANv1.3_b8_fp32.engine",
                        help="TensorRT engine 路径（固定 batch=8）")
    parser.add_argument("--input_dir", default="/workspace/GFPGAN-onnxruntime-demo/temp/detected_faces",
                        help="输入图像目录")
    parser.add_argument("--output_dir", default="/workspace/GFPGAN-onnxruntime-demo/temp/detected_faces-1.3",
                        help="增强后图像目录")
    parser.add_argument("--compare_dir", default="/workspace/GFPGAN-onnxruntime-demo/temp/detected_faces-compare-batch8",
                        help="原图+增强并排对比图目录")
    parser.add_argument("--size", type=int, default=512, help="模型输入尺寸（应与引擎一致）")
    args = parser.parse_args()

    print(f"[CFG] engine={args.engine}")
    print(f"[CFG] input_dir={args.input_dir}")
    print(f"[CFG] output_dir={args.output_dir}")
    print(f"[CFG] compare_dir={args.compare_dir}")

    process_folder(args.engine, args.input_dir, args.output_dir, args.compare_dir, size=args.size)
    print("✅ 完成：增强图与对比图已输出。")

