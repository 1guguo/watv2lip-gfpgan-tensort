#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GFPGAN TensorRT 批处理稳定版（支持分片并行 + 异步写盘 + 进度/ETA）
用法（单进程）:
  python3 01-test-pipeline.py \
    --engine /workspace/GFPGAN-onnxruntime-demo/GFPGANv1.3.engine \
    --input_dir /workspace/GFPGAN-onnxruntime-demo/temp/detected_faces \
    --output_dir /workspace/GFPGAN-onnxruntime-demo/temp/detected_faces-1.3 \
    --compare_dir /workspace/GFPGAN-onnxruntime-demo/temp/detected_faces-compare \
    --jpg_quality 85 \
    --io_threads 3

分片并行（两个进程示例；建议错峰启动）：
  # 分片 0
  CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 OPENCV_OPENCL_RUNTIME=disabled \
  python3 01-test-pipeline.py \
    --engine .../GFPGANv1.3.engine \
    --input_dir .../detected_faces \
    --output_dir .../detected_faces-1.3 \
    --compare_dir .../detected_faces-compare \
    --jpg_quality 85 --io_threads 2 \
    --shard_idx 0 --shard_cnt 2 &

  # 分片 1（错开 0.8s）
  sleep 0.8
  CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 OPENCV_OPENCL_RUNTIME=disabled \
  python3 01-test-pipeline.py \
    --engine .../GFPGANv1.3.engine \
    --input_dir .../detected_faces \
    --output_dir .../detected_faces-1.3 \
    --compare_dir .../detected_faces-compare \
    --jpg_quality 85 --io_threads 2 \
    --shard_idx 1 --shard_cnt 2 &
  wait
"""
import os, sys, time, math, cv2, atexit, argparse, threading, queue
import numpy as np

# === TensorRT / CUDA (不使用 pycuda.autoinit) ===
import tensorrt as trt
import pycuda.driver as cuda

# ---------------- 工具函数 ----------------
def list_images(input_dir):
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    files = [f for f in sorted(os.listdir(input_dir)) if f.lower().endswith(exts)]
    return files

def _safe_mkdir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _safe_imwrite(path, img, jpg_quality=95):
    _safe_mkdir(os.path.dirname(path))
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, int(jpg_quality)])
    else:
        cv2.imwrite(path, img)

def _hstack_compare(orig, enhanced, label_left="original", label_right="enhanced"):
    h, w = orig.shape[:2]
    enh_rs = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LANCZOS4)
    band = max(28, h // 20)
    canvas = np.zeros((h + band, w * 2, 3), dtype=np.uint8)
    canvas[band:, :w] = orig
    canvas[band:, w:] = enh_rs
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, label_left,  (10, band - 8), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(canvas, label_right, (w + 10, band - 8), font, 0.6, (255,255,255), 2, cv2.LINE_AA)
    return canvas

# --------------- 预处理/后处理 ---------------
def preprocess_bgr(img_bgr, size=512):
    """
    GFPGAN 期望 square 输入，这里直接 resize 到 size×size，
    你的 faces 已经是近似 square，避免形变影响（必要时可在生成 faces 阶段裁成正方形）。
    """
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5      # [-1,1]
    img = np.transpose(img, (2, 0, 1))[None, ...].astype(np.float32)  # (1,3,H,W)
    return np.ascontiguousarray(img)

def postprocess_chw(out_chw, target_hw):
    x = np.transpose(out_chw, (1, 2, 0)).astype(np.float32)  # HWC RGB
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

# ----------------- TensorRT Runner -----------------
class TrtRunnerPP:
    """
    稳定版：
    - 每个进程仅创建一个 Runner
    - 只在主线程里调用 run()
    - close() 明确释放顺序，避免 TRTEngine/Context 与 CUDA Context 析构次序导致的驱动报错
    """
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
        self.in_shape = (1,3,size,size)
        self.out_shape = (1,3,size,size)
        self.d_in = [None, None]
        self.d_out = [None, None]
        self.h_in = [None, None]
        self.h_out = [None, None]
        self.bindings = None
        self._warmed = False

        assert os.path.exists(engine_path), f"Engine not found: {engine_path}"

        # 创建 CUDA 上下文（进程内唯一）
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
            self.use_named = (not hasattr(self.engine, "get_binding_name")) and hasattr(self.engine, "get_tensor_name")
            if self.use_named:
                names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
                ins  = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
                outs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
                assert len(ins)==1 and len(outs)==1
                self.in_name, self.out_name = ins[0], outs[0]
                shp_in  = self.engine.get_tensor_shape(self.in_name)
                shp_out = self.engine.get_tensor_shape(self.out_name)
                self.in_dtype  = trt.nptype(self.engine.get_tensor_dtype(self.in_name))
                self.out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.out_name))
            else:
                shp_in  = self.context.get_binding_shape(0)
                shp_out = self.context.get_binding_shape(1)
                self.in_name  = self.engine.get_binding_name(0)
                self.out_name = self.engine.get_binding_name(1)
                self.in_dtype  = trt.nptype(self.engine.get_binding_dtype(0))
                self.out_dtype = trt.nptype(self.engine.get_binding_dtype(1))

            # 动态形状处理
            if -1 in tuple(int(d) if isinstance(d,int) else -1 for d in shp_in):
                self.in_shape = (1,3,size,size)
                if self.use_named:
                    self.context.set_input_shape(self.in_name, self.in_shape)
                else:
                    self.context.set_binding_shape(0, self.in_shape)
            else:
                self.in_shape = tuple(int(x) for x in shp_in)

            if self.use_named:
                shp_out_now = tuple(self.context.get_tensor_shape(self.out_name))
            else:
                shp_out_now = tuple(self.context.get_binding_shape(1))
            if -1 in tuple(int(d) if isinstance(d,int) else -1 for d in shp_out_now):
                self.out_shape = (1,3,self.in_shape[-2], self.in_shape[-1])
            else:
                self.out_shape = tuple(int(x) for x in shp_out_now)

            # 预分配双缓冲
            in_bytes  = int(np.prod(self.in_shape))  * np.dtype(self.in_dtype).itemsize
            out_bytes = int(np.prod(self.out_shape)) * np.dtype(self.out_dtype).itemsize
            for k in (0,1):
                self.d_in[k]  = cuda.mem_alloc(in_bytes)
                self.d_out[k] = cuda.mem_alloc(out_bytes)
                self.h_in[k]  = cuda.pagelocked_empty(self.in_shape,  dtype=self.in_dtype)
                self.h_out[k] = cuda.pagelocked_empty(self.out_shape, dtype=self.out_dtype)

            # 绑定
            if not self.use_named:
                self.bindings = [0]*self.engine.num_bindings

        except Exception:
            self.close()
            raise

        # 确保退出时释放
        atexit.register(self._atexit_close)

    def _atexit_close(self):
        try:
            self.close()
        except:
            pass

    def warmup(self, img_bgr, loops=3):
        if self._warmed:
            return
        net_in = preprocess_bgr(img_bgr, size=self.in_shape[-1]).astype(self.in_dtype, copy=False)
        for _ in range(loops):
            for ping in (0,1):
                self.h_in[ping][...] = net_in
                cuda.memcpy_htod_async(self.d_in[ping], self.h_in[ping], self.stream)
                if self.use_named:
                    self.context.set_tensor_address(self.in_name,  int(self.d_in[ping]))
                    self.context.set_tensor_address(self.out_name, int(self.d_out[ping]))
                    _enqueue_v3(self.context, self.stream)
                else:
                    self.bindings[0] = int(self.d_in[ping])
                    self.bindings[1] = int(self.d_out[ping])
                    if hasattr(self.context, "execute_async_v2"):
                        self.context.execute_async_v2(self.bindings, self.stream.handle)
                    else:
                        self.context.execute_v2(self.bindings)
                cuda.memcpy_dtoh_async(self.h_out[ping], self.d_out[ping], self.stream)
            self.stream.synchronize()
        self._warmed = True

    def run(self, img_bgr: np.ndarray):
        """
        单张推理（主线程调用）
        """
        h0, w0 = img_bgr.shape[:2]
        net_in = preprocess_bgr(img_bgr, size=self.in_shape[-1]).astype(self.in_dtype, copy=False)

        # 双缓冲 ping-pong 一次
        ping = 0
        self.h_in[ping][...] = net_in
        cuda.memcpy_htod_async(self.d_in[ping], self.h_in[ping], self.stream)

        if self.use_named:
            self.context.set_tensor_address(self.in_name,  int(self.d_in[ping]))
            self.context.set_tensor_address(self.out_name, int(self.d_out[ping]))
            _enqueue_v3(self.context, self.stream)
        else:
            self.bindings[0] = int(self.d_in[ping])
            self.bindings[1] = int(self.d_out[ping])
            if hasattr(self.context, "execute_async_v2"):
                self.context.execute_async_v2(self.bindings, self.stream.handle)
            else:
                self.context.execute_v2(self.bindings)

        cuda.memcpy_dtoh_async(self.h_out[ping], self.d_out[ping], self.stream)
        self.stream.synchronize()

        out_img = postprocess_chw(self.h_out[ping][0].astype(np.float32, copy=False), (h0, w0))
        return out_img

    def close(self):
        # 尽量“温柔地”释放，避免驱动层报错
        try:
            if self.stream is not None:
                try: self.stream.synchronize()
                except: pass
        except: pass

        # 解绑 named tensors（TRT 8.5+）
        try:
            if self.use_named and self.context is not None:
                try:
                    self.context.set_tensor_address(self.in_name,  0)
                    self.context.set_tensor_address(self.out_name, 0)
                except: pass
        except: pass

        # 释放 device mem
        try:
            for k in (0,1):
                if self.d_in[k]  is not None:
                    try: self.d_in[k].free()
                    except: pass
                if self.d_out[k] is not None:
                    try: self.d_out[k].free()
                    except: pass
        except: pass

        # 释放 pagelocked host mem
        try:
            for arr in self.h_in + self.h_out:
                try: del arr
                except: pass
        except: pass

        # TRT 句柄
        self.context = None
        self.engine = None
        self.runtime = None

        # CUDA stream
        try:
            if self.stream is not None:
                del self.stream
        except: pass
        self.stream = None

        # 弹出并销毁 CUDA context（最后一步）
        try:
            if self.ctx is not None:
                try: self.ctx.pop()
                except: pass
                try: self.ctx.detach()
                except: pass
        except: pass
        self.ctx = None


# ------------------- 读/写 线程 -------------------
class ReaderThread(threading.Thread):
    """
    仅负责从磁盘读取原图，放进 in_q
    """
    def __init__(self, files, input_dir, in_q, stop_event):
        super().__init__(daemon=True)
        self.files = files
        self.input_dir = input_dir
        self.in_q = in_q
        self.stop_event = stop_event

    def run(self):
        try:
            for idx, fn in enumerate(self.files):
                if self.stop_event.is_set():
                    break
                path = os.path.join(self.input_dir, fn)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                self.in_q.put( (idx, fn, img) )
        finally:
            # 读完后放一个 None 作为终止信号
            self.in_q.put(None)

class WriterThread(threading.Thread):
    """
    仅负责落盘（增强图 + 对比图），从 out_q 取数据
    """
    def __init__(self, out_q, output_dir, compare_dir, jpg_quality, stop_event):
        super().__init__(daemon=True)
        self.out_q = out_q
        self.output_dir = output_dir
        self.compare_dir = compare_dir
        self.jpg_quality = jpg_quality
        self.stop_event = stop_event

    def run(self):
        try:
            while not self.stop_event.is_set():
                item = self.out_q.get()
                if item is None:
                    # 终止信号
                    break
                idx, fn, enhanced_img, orig_img = item
                try:
                    # 写增强图
                    out_path = os.path.join(self.output_dir, fn)
                    _safe_imwrite(out_path, enhanced_img, self.jpg_quality)
                    # 写对比图
                    if self.compare_dir:
                        cmp_img = _hstack_compare(orig_img, enhanced_img, "original", "enhanced")
                        cmp_path = os.path.join(self.compare_dir, os.path.splitext(fn)[0] + "_compare.jpg")
                        _safe_imwrite(cmp_path, cmp_img, 95)
                except Exception as e:
                    # 出错不阻塞整体
                    print(f"[WARN] 写盘失败 {fn}: {e}")
                finally:
                    self.out_q.task_done()
        finally:
            # 确保队列关闭时不会卡住
            pass


# ------------------- 主流程 -------------------
def contiguous_shard(files, shard_idx, shard_cnt):
    if shard_cnt <= 1:
        return files
    n = len(files)
    per = int(math.ceil(n / shard_cnt))
    s = shard_idx * per
    e = min(n, (shard_idx + 1) * per)
    return files[s:e]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="TensorRT engine 路径")
    ap.add_argument("--input_dir", required=True, help="输入图像目录")
    ap.add_argument("--output_dir", required=True, help="增强后图像目录")
    ap.add_argument("--compare_dir", default="", help="原图+增强并排对比图目录（为空则不生成）")
    ap.add_argument("--size", type=int, default=512, help="模型输入尺寸")
    ap.add_argument("--jpg_quality", type=int, default=95, help="JPEG 质量 (1~100)")
    ap.add_argument("--io_threads", type=int, default=2, help="写盘线程数量（读取单线程足够）")
    # 分片设置
    ap.add_argument("--shard_idx", type=int, default=0, help="当前分片索引 [0, shard_cnt)")
    ap.add_argument("--shard_cnt", type=int, default=1, help="分片总数")
    ap.add_argument("--stagger", type=float, default=0.0, help="启动前等待秒数（多进程错峰）")
    args = ap.parse_args()

    print(f"[CFG] engine={args.engine}")
    print(f"[CFG] input_dir={args.input_dir}")
    print(f"[CFG] output_dir={args.output_dir}")
    print(f"[CFG] compare_dir={args.compare_dir}")
    print(f"[CFG] size={args.size}, jpg_quality={args.jpg_quality}, io_threads={args.io_threads}")
    print(f"[CFG] shard={args.shard_idx}/{args.shard_cnt}, stagger={args.stagger}")

    if args.stagger > 0:
        time.sleep(args.stagger)

    assert os.path.exists(args.engine), f"Engine not found: {args.engine}"
    assert os.path.isdir(args.input_dir), f"Input dir not found: {args.input_dir}"
    _safe_mkdir(args.output_dir)
    if args.compare_dir:
        _safe_mkdir(args.compare_dir)

    all_files = list_images(args.input_dir)
    files = contiguous_shard(all_files, args.shard_idx, args.shard_cnt)
    total = len(files)
    print(f"[INFO] files: {total} | io_threads={args.io_threads} | compare={'True' if args.compare_dir else 'False'}")
    if total == 0:
        print("没有需要处理的图片。")
        return

    # 队列
    in_q  = queue.Queue(maxsize=32)   # 读取 -> 推理
    out_q = queue.Queue(maxsize=64)   # 推理 -> 写盘
    stop_event = threading.Event()

    # 线程：读取
    reader = ReaderThread(files, args.input_dir, in_q, stop_event)
    reader.start()

    # 线程：写盘
    writers = []
    for _ in range(max(1, args.io_threads)):
        wt = WriterThread(out_q, args.output_dir, args.compare_dir, args.jpg_quality, stop_event)
        wt.start()
        writers.append(wt)

    runner = None
    processed = 0
    first_img_for_warmup = None
    t0 = time.time()
    sum_infer = 0.0

    try:
        # 先拿第一张做 warmup（更真实的 avg_infer）
        while first_img_for_warmup is None:
            item = in_q.get()
            if item is None:
                break
            idx, fn, img = item
            first_img_for_warmup = img
            # 推回去队列（保持顺序）
            in_q.put(item)
            break

        runner = TrtRunnerPP(args.engine, size=args.size)

        if first_img_for_warmup is not None:
            runner.warmup(first_img_for_warmup, loops=3)

        last_report = t0
        while True:
            item = in_q.get()
            if item is None:
                # 读取结束
                break
            idx, fn, img = item
            in_q.task_done()

            ts = time.time()
            enh = runner.run(img)
            infer_t = time.time() - ts
            sum_infer += infer_t
            processed += 1

            # 送写盘
            out_q.put( (idx, fn, enh, img) )

            # 进度/ETA（每秒一报）
            now = time.time()
            if now - last_report >= 1.0:
                avg_infer = (sum_infer / processed) if processed else 0.0
                elapsed = now - t0
                remain = max(0, total - processed)
                eta = avg_infer * remain
                print(f"[{processed}/{total}] avg_infer={avg_infer:.4f}s  elapsed={elapsed:.1f}s  eta={eta:.1f}s")
                last_report = now

        # 等写盘队列清空
        out_q.join()

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断，准备安全退出…")
    finally:
        # 通知写盘线程退出
        stop_event.set()
        for _ in writers:
            out_q.put(None)
        for wt in writers:
            try: wt.join(timeout=5.0)
            except: pass

        # 关闭 Runner（非常重要）
        if runner is not None:
            runner.close()

    elapsed = time.time() - t0
    avg = (sum_infer / processed) if processed else 0.0
    print("================================================")
    print(f"✅ 完成：{processed}/{total}")
    print(f"⏱ 总耗时: {elapsed:.2f}s   平均每张(推理): {avg:.3f}s")
    print("================================================")


if __name__ == "__main__":
    main()
