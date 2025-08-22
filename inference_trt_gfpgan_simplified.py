import os
import cv2
import subprocess
import platform
import numpy as np
import audio
import argparse
from tqdm import tqdm
from insightface_func.face_detect_crop_single import Face_detect_crop
import time

# --- TensorRT 相关导入 ---
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# -------------------------------
# 参数解析
# -------------------------------
parser = argparse.ArgumentParser(description='使用 TensorRT 的 Wav2Lip 模型进行视频融合')
parser.add_argument('--checkpoint_path', type=str, default='/workspace/wav2lip_256.onnx')
parser.add_argument('--face', type=str, default='/workspace/zhubo.mp4')
parser.add_argument('--audio', type=str, default='/workspace/some6.wav')
parser.add_argument('--outfile', type=str, default='/workspace/results/result_trt2_256.mp4')
parser.add_argument('--static', type=bool, help='如果为 True，则只使用主播视频的第一帧进行推理', default=False)
parser.add_argument('--fps', type=float, help='输出视频的 FPS (默认: 25)', default=25., required=False)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='填充 (top, bottom, left, right)')
parser.add_argument('--face_det_batch_size', type=int, help='人脸检测的批处理大小', default=1)
parser.add_argument('--wav2lip_batch_size', type=int, help='Wav2Lip 模型的批处理大小', default=1)
parser.add_argument('--resize_factor', default=1, type=int, help='通过此因子降低分辨率')
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='裁剪视频到更小区域')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='指定固定的人脸边界框')
parser.add_argument('--rotate', default=False, action='store_true', help='将视频向右旋转90度')
parser.add_argument('--nosmooth', default=False, action='store_true', help='防止在短时间内平滑人脸检测')
parser.add_argument('--preview', default=False, action='store_true', help='推理时预览')
parser.add_argument('--engine_path', type=str, help='TensorRT engine 文件路径', default='/workspace/wav2lip_256.engine')
parser.add_argument('--gfpgan', default=True, action='store_true', help='使用 GFPGAN 增强面部')
parser.add_argument('--gfpgan_engine_path', type=str, help='GFPGAN TensorRT engine 文件路径', default='/workspace/GFPGAN-onnxruntime-demo/gan_512_v1_0625.trt')

args = parser.parse_args()

# 设置图像尺寸为256
if args.checkpoint_path == '/workspace/wav2lip_256.onnx' or args.checkpoint_path == 'checkpoints/wav2lip_256_fp16.onnx':
    args.img_size = 256
else:
    args.img_size = 96

# 确保输出目录存在
os.makedirs(os.path.dirname(args.outfile) if os.path.dirname(args.outfile) else '.', exist_ok=True)
os.makedirs('temp', exist_ok=True)

# -------------------------------
# TensorRT 推理类 (Wav2Lip)
# -------------------------------
class Wav2LipTRT:
    def __init__(self, engine_path, max_batch_size):
        self.max_batch_size = max_batch_size
        self.logger = trt.Logger(trt.Logger.ERROR)
        self.buffers = []
        
        # 直接加载已存在的 TensorRT 引擎
        print(f"加载现有的 TensorRT 引擎 {engine_path}")
        
        # 检查引擎文件是否存在
        if not os.path.exists(engine_path):
            raise RuntimeError(f"TensorRT 引擎文件不存在: {engine_path}")
            
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()

        # 检查TensorRT版本
        try:
            n_io = self.engine.num_io_tensors
            self.tensor_names = [self.engine.get_tensor_name(i) for i in range(n_io)]
            self.is_trt_85 = True
        except AttributeError:
            n_io = self.engine.num_bindings
            self.tensor_names = [self.engine.get_binding_name(i) for i in range(n_io)]
            self.is_trt_85 = False

        print("=== TensorRT Engine I/O Tensors ===")
        for name in self.tensor_names:
            if self.is_trt_85:
                shape = self.context.get_tensor_shape(name)
            else:
                binding_idx = self.engine.get_binding_index(name)
                shape = self.context.get_binding_shape(binding_idx)
            dtype = self.engine.get_tensor_dtype(name) if self.is_trt_85 else self.engine.get_binding_dtype(binding_idx)
            print(f"{name} | Shape: {shape} | Dtype: {dtype}")
        print("==================================")

        # 定义输入输出名称
        if self.is_trt_85:
            # TRT 8.5+ 方式获取输入输出名称
            self.input_names = []
            self.output_names = []
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                tensor_mode = self.engine.get_tensor_mode(tensor_name)
                if tensor_mode == trt.TensorIOMode.INPUT:
                    self.input_names.append(tensor_name)
                elif tensor_mode == trt.TensorIOMode.OUTPUT:
                    self.output_names.append(tensor_name)
        else:
            # 根据ONNX模型结构，使用正确的输入输出名称
            self.input_names = ['mel_spectrogram', 'video_frames']
            self.output_names = ['predicted_frames']
            
        self.stream = cuda.Stream()

        # 预分配最大批处理大小的内存
        self.host_inputs = {}
        self.device_inputs = {}
        self.host_outputs = {}
        self.device_outputs = {}

        for name in self.tensor_names:
            if self.is_trt_85:
                shape = self.context.get_tensor_shape(name)
                dtype = self.engine.get_tensor_dtype(name)
            else:
                binding_idx = self.engine.get_binding_index(name)
                shape = self.context.get_binding_shape(binding_idx)
                dtype = self.engine.get_binding_dtype(binding_idx)

            if dtype == trt.float32:
                numpy_dtype = np.float32
            elif dtype == trt.int32:
                numpy_dtype = np.int32
            else:
                raise TypeError(f"Unsupported data type {dtype}")

            # 为最大批处理大小预分配内存
            max_shape = list(shape)
            max_shape[0] = self.max_batch_size
            max_shape = tuple(max_shape)
            
            size = np.prod(max_shape)
            host_mem = np.empty(max_shape, dtype=numpy_dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            if name in self.input_names:
                self.host_inputs[name] = host_mem
                self.device_inputs[name] = device_mem
            elif name in self.output_names:
                self.host_outputs[name] = host_mem
                self.device_outputs[name] = device_mem
            else:
                 print(f"[Warning] Unknown tensor name: {name}")

        # 显式绑定输入输出（TRT 8.5+ 需要 set_tensor_address）
        if self.is_trt_85:
            for name, mem in self.device_inputs.items():
                # TRT 8.5+ 不再需要在这里设置输入形状，而是在推理时设置
                self.context.set_tensor_address(name, int(mem))
            for name, mem in self.device_outputs.items():
                self.context.set_tensor_address(name, int(mem))

    def run_inference(self, video_frames_nhwc, mel_spectrogram_nhwc):
        """执行推理 (使用预分配内存)"""
        batch_size = video_frames_nhwc.shape[0]
        
        # 检查批处理大小是否超出预分配范围
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size ({batch_size}) exceeds pre-allocated max batch size ({self.max_batch_size})")

        # 准备输入数据 (NCHW) - 只拷贝当前批次的数据到预分配的缓冲区
        video_input_nchw = np.ascontiguousarray(video_frames_nhwc.transpose(0, 3, 1, 2))
        mel_input_nchw = np.ascontiguousarray(mel_spectrogram_nhwc.transpose(0, 3, 1, 2))
        
        # 根据ONNX模型结构，使用正确的输入名称
        mel_input_name = 'mel_spectrogram'
        video_input_name = 'video_frames'
        output_name = 'predicted_frames'
            
        # 将数据拷贝到缓冲区视图中
        np.copyto(self.host_inputs[mel_input_name][:batch_size], mel_input_nchw)
        np.copyto(self.host_inputs[video_input_name][:batch_size], video_input_nchw)

        # Host -> Device - 只传输当前批次大小的数据
        mel_device_mem = self.device_inputs[mel_input_name]
        video_device_mem = self.device_inputs[video_input_name]
        mel_host_mem_view = self.host_inputs[mel_input_name][:batch_size]
        video_host_mem_view = self.host_inputs[video_input_name][:batch_size]
        
        cuda.memcpy_htod_async(mel_device_mem, mel_host_mem_view, self.stream)
        cuda.memcpy_htod_async(video_device_mem, video_host_mem_view, self.stream)

        # 为 TRT < 8.5 设置动态绑定
        if not self.is_trt_85:
            # 设置输入张量的形状
            self.context.set_binding_shape(self.engine.get_binding_index(mel_input_name), mel_input_nchw.shape)
            self.context.set_binding_shape(self.engine.get_binding_index(video_input_name), video_input_nchw.shape)
            
            # 准备 bindings 列表
            bindings = [None] * self.engine.num_bindings
            bindings[self.engine.get_binding_index(mel_input_name)] = int(mel_device_mem)
            bindings[self.engine.get_binding_index(video_input_name)] = int(video_device_mem)
            bindings[self.engine.get_binding_index(output_name)] = int(self.device_outputs[output_name])
        else:
            # TRT 8.5+ 设置输入形状
            self.context.set_input_shape(mel_input_name, mel_input_nchw.shape)
            self.context.set_input_shape(video_input_name, video_input_nchw.shape)

        # 执行推理
        if self.is_trt_85:
            self.context.execute_async_v3(self.stream.handle)
        else:
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.handle)

        # Device -> Host - 只传输当前批次大小的输出数据
        if not self.is_trt_85:
             output_shape = self.context.get_binding_shape(self.engine.get_binding_index(output_name))
        else:
            output_shape = self.context.get_tensor_shape(output_name)
             
        # 创建输出 host buffer 的视图
        output_host_view = self.host_outputs[output_name][:batch_size] 
        
        cuda.memcpy_dtoh_async(output_host_view, self.device_outputs[output_name], self.stream)
        self.stream.synchronize()

        # 处理输出 - 使用视图
        output = output_host_view
        output = output.transpose(0, 2, 3, 1)  # → [B, H, W, 3]
        output = np.clip(output, 0, 1) * 255    # 反归一化到 0~255
        output = output.astype(np.uint8)
        return output

    def __del__(self):
        try:
            if hasattr(self, 'stream'):
                try:
                    self.stream.synchronize()
                except:
                    pass
            # 这些对象交给 Python 回收即可；若你手动mem_alloc，记得放进 self.buffers 统一释放
            if hasattr(self, 'context'):
                del self.context
            if hasattr(self, 'engine'):
                del self.engine
            if hasattr(self, 'stream'):
                del self.stream
            if hasattr(self, 'buffers'):
                for buf in self.buffers:
                    try:
                        del buf
                    except:
                        pass
        except Exception as e:
            print(f"清理 Wav2LipTRT 资源时出错: {e}")

# -------------------------------
# GFPGAN TensorRT 增强类
# -------------------------------
class GFPGANTRT:
    def __init__(self, engine_path):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    # 分配输入输出缓冲区
    def allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            shape = engine.get_tensor_shape(name)
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append((name, host_mem, device_mem))
            else:
                outputs.append((name, host_mem, device_mem))
                
        return inputs, outputs, bindings, stream

    # 推理过程
    def do_inference(self, bindings, inputs, outputs, stream):
        for name, host_mem, device_mem in inputs:
            cuda.memcpy_htod_async(device_mem, host_mem, stream)
            self.context.set_tensor_address(name, device_mem)
            
        for name, host_mem, device_mem in outputs:
            self.context.set_tensor_address(name, device_mem)
        
        self.context.execute_async_v3(stream_handle=stream.handle)
        
        for name, host_mem, device_mem in outputs:
            cuda.memcpy_dtoh_async(host_mem, device_mem, stream)
            
        stream.synchronize()
        return outputs[0][1]  # 返回第一个输出的 host_mem

    # 图像预处理：BGR -> RGB, normalize [-1, 1], NCHW
    def preprocess_image(self, image):
        img = cv2.resize(image, (512, 512))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # [-1, 1]
        img = img.transpose(2, 0, 1)[None, ...]  # NCHW
        return img.astype(np.float32)

    # 图像后处理：NCHW -> HWC, [0, 255], RGB -> BGR
    def postprocess_output(self, output_array):
        img = output_array.reshape(3, 512, 512).transpose(1, 2, 0)
        img = ((img + 1) * 0.5 * 255).clip(0, 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img

    def enhance(self, img):
        """
        使用 GFPGAN 增强图像
        输入: BGR 格式的图像 (H, W, C)
        输出: BGR 格式的增强图像 (H, W, C)
        """
        input_img = self.preprocess_image(img)
        np.copyto(self.inputs[0][1], input_img.ravel())
        
        output_host_mem = self.do_inference(self.bindings, self.inputs, self.outputs, self.stream)
        out_img = self.postprocess_output(output_host_mem)
        
        # 调整回原始尺寸
        out_img = cv2.resize(out_img, (img.shape[1], img.shape[0]))
        return out_img

    def __del__(self):
        # 清理资源
        if hasattr(self, 'context'): del self.context
        if hasattr(self, 'engine'): del self.engine

# -------------------------------
# 工具函数
# -------------------------------
def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    detector = Face_detect_crop(name='antelope', root='/workspace/wav2lip-onnx-256/insightface_func/models')
    detector.prepare(ctx_id=0, det_thresh=0.3, det_size=(320, 320), mode='none')
    predictions = []
    for i in tqdm(range(0, len(images)), desc="Detecting faces"):
        bbox = detector.getBox(images[i])
        predictions.append(bbox)
    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')
        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        # 确保裁剪区域是正方形，便于后续处理
        crop_width = x2 - x1
        crop_height = y2 - y1
        
        # 取宽高的最大值作为正方形的边长
        side_length = max(crop_width, crop_height)
        
        # 计算新的边界坐标，使裁剪区域成为正方形
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # 确保正方形区域不超出图像边界
        half_side = side_length // 2
        new_x1 = max(0, center_x - half_side)
        new_x2 = min(image.shape[1], new_x1 + side_length)
        new_y1 = max(0, center_y - half_side)
        new_y2 = min(image.shape[0], new_y1 + side_length)
        
        # 如果计算后的坐标导致尺寸不对，重新调整
        actual_width = new_x2 - new_x1
        actual_height = new_y2 - new_y1
        if actual_width != actual_height:
            side_length = min(actual_width, actual_height)
            new_x2 = new_x1 + side_length
            new_y2 = new_y1 + side_length
            
        results.append([x1, y1, x2, y2, new_x1, new_y1, new_x2, new_y2])
    boxes = np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2, _, _, _, _) in results])
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    # 返回原始裁剪区域和正方形裁剪区域
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2), image[new_y1: new_y2, new_x1:new_x2], (new_y1, new_y2, new_x1, new_x2)] 
               for image, (x1, y1, x2, y2, new_x1, new_y1, new_x2, new_y2) in zip(images, results)]
    del detector
    return results

def load_video_frames(video_path):
    """加载视频帧"""
    if not os.path.isfile(video_path):
        raise ValueError(f'视频文件不存在: {video_path}')
    
    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    print(f'读取视频 {video_path} (FPS: {fps})...')
    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        if args.resize_factor > 1:
            frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))
        if args.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        y1, y2, x1, x2 = args.crop
        if x2 == -1: x2 = frame.shape[1]
        if y2 == -1: y2 = frame.shape[0]
        frame = frame[y1:y2, x1:x2]
        frames.append(frame)
    print(f"共读取 {len(frames)} 帧.")
    return frames, fps

def extract_audio_to_wav(input_path, output_wav_path='temp/audio.wav'):
    """从视频或音频文件中提取或转换音频为 WAV"""
    print('提取或转换音频...')
    command = f'ffmpeg -y -i "{input_path}" -ac 1 -ar 16000 -acodec pcm_s16le "{output_wav_path}" -hide_banner -loglevel error'
    subprocess.call(command, shell=True)
    return output_wav_path

def datagen(face_frames, mels):
    """数据生成器，用于批处理"""
    img_batch, mel_batch, frame_batch, coords_batch, square_faces_batch = [], [], [], [], []
    
    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(face_frames)
        else:
            face_det_results = face_detect([face_frames[0]])
    else:
        print('使用指定的边界框...')
        y1, y2, x1, x2 = args.box
        # 对于指定边界框，也生成正方形裁剪区域
        face_det_results = []
        for f in face_frames:
            crop_width = x2 - x1
            crop_height = y2 - y1
            side_length = max(crop_width, crop_height)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            half_side = side_length // 2
            new_x1 = max(0, center_x - half_side)
            new_x2 = min(f.shape[1], new_x1 + side_length)
            new_y1 = max(0, center_y - half_side)
            new_y2 = min(f.shape[0], new_y1 + side_length)
            
            if new_x2 - new_x1 != new_y2 - new_y1:
                side_length = min(new_x2 - new_x1, new_y2 - new_y1)
                new_x2 = new_x1 + side_length
                new_y2 = new_y1 + side_length
                
            face_det_results.append([f[y1: y2, x1:x2], (y1, y2, x1, x2), f[new_y1: new_y2, new_x1:new_x2], (new_y1, new_y2, new_x1, new_x2)])

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(face_frames)
        
        frame_to_save = face_frames[idx].copy()
        face, coords, square_face, square_coords = face_det_results[idx].copy()
        face = cv2.resize(face, (args.img_size, args.img_size))
        # 将正方形人脸也调整为模型输入尺寸
        square_face = cv2.resize(square_face, (args.img_size, args.img_size))
        
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        square_faces_batch.append(square_face)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch_np = np.asarray(img_batch)
            mel_batch_np = np.asarray(mel_batch)
            
            img_masked = img_batch_np.copy()
            img_masked[:, args.img_size//2:] = 0
            img_batch_processed = np.concatenate((img_masked, img_batch_np), axis=3) / 255.
            mel_batch_processed = np.reshape(mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1])
            
            yield img_batch_processed, mel_batch_processed, frame_batch, coords_batch, square_faces_batch
            
            img_batch, mel_batch, frame_batch, coords_batch, square_faces_batch = [], [], [], [], []

    if len(img_batch) > 0:
        img_batch_np = np.asarray(img_batch)
        mel_batch_np = np.asarray(mel_batch)
        
        img_masked = img_batch_np.copy()
        img_masked[:, args.img_size//2:] = 0
        img_batch_processed = np.concatenate((img_masked, img_batch_np), axis=3) / 255.
        mel_batch_processed = np.reshape(mel_batch_np, [len(mel_batch_np), mel_batch_np.shape[1], mel_batch_np.shape[2], 1])
        
        yield img_batch_processed, mel_batch_processed, frame_batch, coords_batch, square_faces_batch

# -------------------------------
# 主函数
# -------------------------------
def main():
    start_time = time.time()
    print("开始处理...")

    print("--- 加载主播视频帧 ---")
    load_video_start = time.time()
    face_frames, face_fps = load_video_frames(args.face)
    load_video_time = time.time() - load_video_start
    print(f"加载视频耗时: {load_video_time:.2f} 秒")

    print("--- 提取音频 ---")
    extract_audio_start = time.time()
    temp_audio_path = extract_audio_to_wav(args.audio, 'temp/temp_audio.wav')
    extract_audio_time = time.time() - extract_audio_start
    print(f"提取音频耗时: {extract_audio_time:.2f} 秒")

    print("--- 处理音频 ---")
    process_audio_start = time.time()
    wav = audio.load_wav(temp_audio_path, 16000)
    mel = audio.melspectrogram(wav)
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
    process_audio_time = time.time() - process_audio_start
    print(f"处理音频耗时: {process_audio_time:.2f} 秒")

    mel_step_size = 16
    fps = args.fps
    mel_chunks = []
    mel_idx_multiplier = 80./fps 
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    print(f"生成了 {len(mel_chunks)} 个 mel chunks.")

    print("--- 匹配视频帧与音频块 ---")
    match_frames_start = time.time()
    num_mel_chunks = len(mel_chunks)
    print(f"需要处理 {num_mel_chunks} 个帧.")
    if len(face_frames) < num_mel_chunks:
        print("主播视频帧数不足，将循环使用帧。")
        extended_frames = []
        for i in range(num_mel_chunks):
            idx = i % len(face_frames) if not args.static else 0
            extended_frames.append(face_frames[idx])
        face_frames = extended_frames
    elif len(face_frames) > num_mel_chunks:
        print("主播视频帧数过多，将截断。")
        face_frames = face_frames[:num_mel_chunks]
    
    print(f"最终用于处理的主播视频帧数: {len(face_frames)}")
    match_frames_time = time.time() - match_frames_start
    print(f"匹配视频帧与音频块耗时: {match_frames_time:.2f} 秒")

    print("--- 初始化 TensorRT 模型 ---")
    init_model_start = time.time()
    model = Wav2LipTRT(args.engine_path, max_batch_size=args.wav2lip_batch_size)
    
    # 初始化 GFPGAN 增强器（如果启用）
    gfpgan = None
    if args.gfpgan:
        try:
            print("--- 初始化 GFPGAN 增强器 ---")
            gfpgan = GFPGANTRT(args.gfpgan_engine_path)
        except Exception as e:
            print(f"警告: 无法初始化 GFPGAN 增强器: {e}")
            args.gfpgan = False
    init_model_time = time.time() - init_model_start
    print(f"初始化模型耗时: {init_model_time:.2f} 秒")

    frame_h, frame_w = face_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    print("--- 开始推理和视频合成 ---")
    inference_start_time = time.time()
    batch_size = args.wav2lip_batch_size
    gen = datagen(face_frames.copy(), mel_chunks)
    
    # 性能统计变量
    total_frames = 0
    wav2lip_inference_time = 0
    gfpgan_enhance_time = 0
    video_write_time = 0
    
    pbar = tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)), desc="Processing batches")
    for i, (img_batch, mel_batch, frames, coords, square_faces) in enumerate(pbar):
        batch_start_time = time.time()
        # img_batch: [B, H, W, 6] (float32, 0-1)
        # mel_batch: [B, 80, 16, 1] (float32)
        
        wav2lip_start = time.time()
        pred = model.run_inference(img_batch, mel_batch) # Output: [B, H, W, 3] (uint8)
        wav2lip_inference_time += time.time() - wav2lip_start
        
        batch_process_start = time.time()
        for p, f, c, square_face in zip(pred, frames, coords, square_faces):
            y1, y2, x1, x2 = c
            p_resized = cv2.resize(p, (x2 - x1, y2 - y1))
            
            # 如果启用了 GFPGAN 增强，则对人脸区域进行增强
            if args.gfpgan and gfpgan:
                try:
                    gfpgan_start = time.time()
                    # 使用正方形人脸图像进行增强
                    # 将Wav2Lip输出的人脸图像调整为正方形以匹配原始人脸比例
                    p_square = cv2.resize(p_resized, (square_face.shape[1], square_face.shape[0]))
                    enhanced_face = gfpgan.enhance(p_square)
                    # 将增强后的人脸图像调整回原始尺寸
                    enhanced_face_resized = cv2.resize(enhanced_face, (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2] = enhanced_face_resized
                    gfpgan_enhance_time += time.time() - gfpgan_start
                except Exception as e:
                    print(f"警告: GFPGAN 增强失败，使用原始人脸: {e}")
                    f[y1:y2, x1:x2] = p_resized
            else:
                f[y1:y2, x1:x2] = p_resized
            
            write_start = time.time()
            out.write(f)
            video_write_time += time.time() - write_start
            
            total_frames += 1
            if args.preview:
                cv2.imshow("Fusion Result", f)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        batch_process_time = time.time() - batch_start_time
    
    out.release()
    if args.preview:
        cv2.destroyAllWindows()
    
    inference_time = time.time() - inference_start_time
    
    print("--- 合成最终视频 ---")
    video_synthesis_start = time.time()
    command = f'ffmpeg -y -i "{temp_audio_path}" -i temp/result.avi -c:v copy -c:a aac -strict experimental "{args.outfile}" -hide_banner -loglevel error'
    subprocess.call(command, shell=platform.system() != 'Windows')
    video_synthesis_time = time.time() - video_synthesis_start
    print(f"视频合成耗时: {video_synthesis_time:.2f} 秒")

    try:
        os.remove('temp/result.avi')
        os.remove(temp_audio_path)
    except OSError:
        pass

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = elapsed_time % 60
    
    # 输出详细的性能统计信息
    print("\n=== 性能统计 ===")
    print(f"总耗时: {hours:02d}小时:{minutes:02d}分钟:{seconds:.2f}秒 ({elapsed_time:.2f} 秒)")
    print(f"加载视频耗时: {load_video_time:.2f} 秒")
    print(f"提取音频耗时: {extract_audio_time:.2f} 秒")
    print(f"处理音频耗时: {process_audio_time:.2f} 秒")
    print(f"匹配视频帧与音频块耗时: {match_frames_time:.2f} 秒")
    print(f"初始化模型耗时: {init_model_time:.2f} 秒")
    print(f"推理和视频合成耗时: {inference_time:.2f} 秒")
    print(f"  Wav2Lip推理耗时: {wav2lip_inference_time:.2f} 秒")
    if args.gfpgan:
        print(f"  GFPGAN增强耗时: {gfpgan_enhance_time:.2f} 秒")
    print(f"  视频写入耗时: {video_write_time:.2f} 秒")
    print(f"视频合成耗时: {video_synthesis_time:.2f} 秒")
    
    if total_frames > 0:
        avg_time_per_frame = elapsed_time / total_frames
        fps = total_frames / elapsed_time
        print(f"\n处理帧数: {total_frames}")
        print(f"平均每帧处理时间: {avg_time_per_frame:.4f} 秒")
        print(f"处理速度: {fps:.2f} FPS")
        
        avg_wav2lip_time = wav2lip_inference_time / total_frames
        print(f"平均每帧Wav2Lip推理时间: {avg_wav2lip_time:.4f} 秒")
        
        if args.gfpgan and gfpgan:
            avg_gfpgan_time = gfpgan_enhance_time / total_frames
            print(f"平均每帧GFPGAN增强时间: {avg_gfpgan_time:.4f} 秒")
        
        avg_write_time = video_write_time / total_frames
        print(f"平均每帧视频写入时间: {avg_write_time:.4f} 秒")
    
    print(f"\n输出视频已保存至: {args.outfile}")

if __name__ == '__main__':
    main()