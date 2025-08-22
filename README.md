# Wav2Lip with GFPGAN Enhancement (TensorRT Version)

This project implements a high-performance lip-syncing solution using Wav2Lip and GFPGAN models with TensorRT acceleration. It takes a video and an audio file as input and generates a lip-synced video with enhanced face quality.

## Features

- **High Performance**: Uses TensorRT for accelerated inference
- **Face Enhancement**: Integrates GFPGAN for face restoration and enhancement
- **Batch Processing**: Supports batch processing for improved performance
- **Performance Statistics**: Provides detailed timing information for each processing step
- **Flexible Configuration**: Multiple parameters for customization

## Prerequisites

Ensure you have the following files:
- Input video file (mp4, flv, avi)
- Input audio file (mp3, wav)
- Wav2Lip TensorRT engine file
- GFPGAN TensorRT engine file

## Installation

```bash
# Install required dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python inference_trt_gfpgan_simplified.py \
  --face /workspace/zhubo.mp4 \
  --audio /workspace/some6.wav \
  --outfile /workspace/results/result_trt2_256.mp4
```

### Advanced Usage

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

### Disable GFPGAN Enhancement

```bash
python inference_trt_gfpgan_simplified.py \
  --face /workspace/zhubo.mp4 \
  --audio /workspace/some6.wav \
  --outfile /workspace/results/result_trt2_256.mp4 \
  --gfpgan False
```

### Preview During Processing

```bash
python inference_trt_gfpgan_simplified.py \
  --face /workspace/zhubo.mp4 \
  --audio /workspace/some6.wav \
  --outfile /workspace/results/result_trt2_256.mp4 \
  --preview
```

## Parameters

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

## Performance Statistics

The application provides detailed performance statistics including:
- Total processing time
- Time for each processing step
- Average time per frame
- Processing speed (FPS)

Example output:
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

## API Service

We also provide Flask and FastAPI services for web-based inference:

### Flask Service

```bash
python flask_trt_simplified.py
```

### FastAPI Service

```bash
python fastapi_trt_simplified.py
```

Or with uvicorn:

```bash
uvicorn fastapi_trt_simplified:app --host 0.0.0.0 --port 120
```

## API Endpoints

- `GET /` - Health check
- `GET /file/{filename}` - Get generated video file
- `POST /wav2lip` - Perform lip-syncing with face enhancement

For POST `/wav2lip`, send a multipart form with:
- `face`: Video file (mp4, flv, avi)
- `audio`: Audio file (mp3, wav)

## Troubleshooting

1. **Black face region in output**: Ensure GFPGAN engine file is correctly loaded and compatible
2. **Slow processing**: Try increasing `--wav2lip_batch_size` if you have sufficient GPU memory
3. **Memory issues**: Reduce batch size or use `--resize_factor` to lower resolution
4. **Face not detected**: Adjust `--pads` parameter to include more context around the face

## License

This project is licensed under the MIT License - see the LICENSE file for details.
