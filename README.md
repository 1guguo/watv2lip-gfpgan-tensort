# Wav2Lip with GFPGAN Enhancement (TensorRT Version)
# Wav2Lip 与 GFPGAN 增强 (TensorRT 版本)

This project implements a high-performance lip-syncing solution using Wav2Lip and GFPGAN models with TensorRT acceleration. It takes a video and an audio file as input and generates a lip-synced video with enhanced face quality.
本项目实现了一个使用 Wav2Lip 和 GFPGAN 模型的高性能唇同步解决方案，采用 TensorRT 加速。它以视频和音频文件作为输入，生成唇同步视频，并增强面部质量。

## Features / 功能特性

- **High Performance**: Uses TensorRT for accelerated inference
- **Face Enhancement**: Integrates GFPGAN for face restoration and enhancement
- **Batch Processing**: Supports batch processing for improved performance
- **Performance Statistics**: Provides detailed timing information for each processing step
- **Flexible Configuration**: Multiple parameters for customization
- **高性能**: 使用 TensorRT 进行加速推理
- **面部增强**: 集成 GFPGAN 进行面部修复和增强
- **批量处理**: 支持批量处理以提高性能
- **性能统计**: 提供每个处理步骤的详细时间信息
- **灵活配置**: 多种参数可自定义

## Prerequisites / 先决条件

Ensure you have the following files:
确保您有以下文件：
- Input video file (mp4, flv, avi)
- Input audio file (mp3, wav)
- Wav2Lip TensorRT engine file
- GFPGAN TensorRT engine file
- 输入视频文件 (mp4, flv, avi)
- 输入音频文件 (mp3, wav)
- Wav2Lip TensorRT 引擎文件
- GFPGAN TensorRT 引擎文件

## Installation / 安装

```bash
# Install required dependencies
# 安装所需依赖
pip install -r requirements.txt
```

## Usage / 使用方法

### Basic Usage / 基本用法

```bash
python inference_trt_gfpgan_simplified.py \
  --face /workspace/zhubo.mp4 \
  --audio /workspace/some6.wav \
  --outfile /workspace/results/result_trt2_256.mp4
```

### Advanced Usage / 高级用法

```bash
python inference_trt_gfpgan_simplified.py \
  --face /workspace/zhubo.mp4 \
  --audio /workspace/some6.wav \
  --outfile /workspace/results/result_trt2_256.mp4 \
  --engine_path /workspace/wav2lip_256.engine \
  --gfpgan_engine_path /workspace/GFPGANv1.3.engine \
  --wav2lip_batch_size 16 \
  --fps 30 \
  --pads 0 10 0 0 \
  --resize_factor 2
```

### Disable GFPGAN Enhancement / 禁用 GFPGAN 增强

```bash
python inference_trt_gfpgan_simplified.py \
  --face /workspace/zhubo.mp4 \
  --audio /workspace/some6.wav \
  --outfile /workspace/results/result_trt2_256.mp4 \
  --gfpgan False
```

### Preview During Processing / 处理过程中预览

```bash
python inference_trt_gfpgan_simplified.py \
  --face /workspace/zhubo.mp4 \
  --audio /workspace/some6.wav \
  --outfile /workspace/results/result_trt2_256.mp4 \
  --preview
```

## Parameters / 参数说明

| Parameter | Description | Default |
|----------|-------------|---------|
| `--face` | Path to the input video file | `/workspace/zhubo.mp4` |
| `--audio` | Path to the input audio file | `/workspace/some6.wav` |
| `--outfile` | Path to the output video file | `/workspace/results/result_trt2_256.mp4` |
| `--engine_path` | Path to the Wav2Lip TensorRT engine | `/workspace/wav2lip_256.engine` |
| `--gfpgan_engine_path` | Path to the GFPGAN TensorRT engine | `/workspace/GFPGAN-onnxruntime-demo/gan_512_v1_0625.trt` |
| `--wav2lip_batch_size` | Batch size for Wav2Lip inference | `1` |
| `--fps` | Output video FPS | `25.0` |
| `--pads` | Padding for face detection (top, bottom, left, right) | `0 10 0 0` |
| `--resize_factor` | Factor to resize the video | `1` |
| `--gfpgan` | Enable/disable GFPGAN enhancement | `True` |
| `--preview` | Preview the output during processing | `False` |
| `--static` | Use only the first frame of the video | `False` |

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--face` | 输入视频文件路径 | `/workspace/zhubo.mp4` |
| `--audio` | 输入音频文件路径 | `/workspace/some6.wav` |
| `--outfile` | 输出视频文件路径 | `/workspace/results/result_trt2_256.mp4` |
| `--engine_path` | Wav2Lip TensorRT 引擎路径 | `/workspace/wav2lip_256.engine` |
| `--gfpgan_engine_path` | GFPGAN TensorRT 引擎路径 | `/workspace/GFPGAN-onnxruntime-demo/gan_512_v1_0625.trt` |
| `--wav2lip_batch_size` | Wav2Lip 推理的批处理大小 | `1` |
| `--fps` | 输出视频 FPS | `25.0` |
| `--pads` | 人脸检测填充 (上, 下, 左, 右) | `0 10 0 0` |
| `--resize_factor` | 视频缩放因子 | `1` |
| `--gfpgan` | 启用/禁用 GFPGAN 增强 | `True` |
| `--preview` | 处理过程中预览输出 | `False` |
| `--static` | 仅使用视频的第一帧 | `False` |

## Performance Statistics / 性能统计

The application provides detailed performance statistics including:
应用程序提供详细的性能统计信息，包括：
- Total processing time / 总处理时间
- Time for each processing step / 每个处理步骤的时间
- Average time per frame / 平均每帧时间
- Processing speed (FPS) / 处理速度 (FPS)

Example output:
输出示例：
```
=== 性能统计 ===
总耗时: 00小时:00分钟:51.54秒 (51.54 秒)
加载视频耗时: 1.70 秒
提取音频耗时: 0.02 秒
处理音频耗时: 0.44 秒
匹配视频帧与音频块耗时: 0.00 秒
初始化模型耗时: 0.65 秒
推理和视频合成耗时: 48.66 秒
  Wav2Lip推理耗时: 3.00 秒
  GFPGAN增强耗时: 35.97 秒
  视频写入耗时: 4.34 秒
视频合成耗时: 0.06 秒

处理帧数: 586
平均每帧处理时间: 0.0879 秒
处理速度: 11.37 FPS
平均每帧Wav2Lip推理时间: 0.0051 秒
平均每帧GFPGAN增强时间: 0.0614 秒
平均每帧视频写入时间: 0.0074 秒
```

## API Service / API 服务

We also provide Flask and FastAPI services for web-based inference:
我们还提供 Flask 和 FastAPI 服务用于基于 Web 的推理：

### Flask Service / Flask 服务

```bash
python flask_trt_simplified.py
```

### FastAPI Service / FastAPI 服务

```bash
python fastapi_trt_simplified.py
```

Or with uvicorn:
或者使用 uvicorn：

```bash
uvicorn fastapi_trt_simplified:app --host 0.0.0.0 --port 120
```

## API Endpoints / API 端点

- `GET /` - Health check / 健康检查
- `GET /file/{filename}` - Get generated video file / 获取生成的视频文件
- `POST /wav2lip` - Perform lip-syncing with face enhancement / 执行唇同步和面部增强

For POST `/wav2lip`, send a multipart form with:
对于 POST `/wav2lip`，发送包含以下内容的 multipart 表单：
- `face`: Video file (mp4, flv, avi) / 视频文件
- `audio`: Audio file (mp3, wav) / 音频文件

## Troubleshooting / 故障排除

1. **Black face region in output**: Ensure GFPGAN engine file is correctly loaded and compatible
2. **Slow processing**: Try increasing `--wav2lip_batch_size` if you have sufficient GPU memory
3. **Memory issues**: Reduce batch size or use `--resize_factor` to lower resolution
4. **Face not detected**: Adjust `--pads` parameter to include more context around the face
1. **输出中人脸区域变黑**: 确保 GFPGAN 引擎文件正确加载且兼容
2. **处理缓慢**: 如果 GPU 内存充足，尝试增加 `--wav2lip_batch_size`
3. **内存问题**: 减少批处理大小或使用 `--resize_factor` 降低分辨率
4. **未检测到人脸**: 调整 `--pads` 参数以包含更多面部周围上下文

## License / 许可证

This project is licensed under the MIT License - see the LICENSE file for details.
该项目采用 MIT 许可证 - 详情请见 LICENSE 文件。
