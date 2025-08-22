from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
import tempfile
import logging
from flask_cors import CORS
from datetime import datetime
import shutil

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

# 输出目录
OUTPUT_DIR = "/workspace/result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/", methods=['GET'])
def test1():
    return "Wav2Lip GFPGAN Flask API"

@app.route('/file/<filename>')
def get_video(filename):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"message": "File not found", "status": 404}), 404
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/wav2lip', methods=['POST'])
def inference():
    logging.info("Received request for inference")

    video_file = request.files.get('face')
    audio_file = request.files.get('audio')

    if not video_file:
        return jsonify({"message": "No video file", "status": 400}), 400
    if video_file.filename.rsplit('.', 1)[-1].lower() not in {'mp4', 'flv', 'avi'}:
        return jsonify({"message": "Video format should be mp4/flv/avi", "status": 400}), 400

    if not audio_file:
        return jsonify({"message": "No audio file", "status": 400}), 400
    if audio_file.filename.rsplit('.', 1)[-1].lower() not in {'mp3', 'wav'}:
        return jsonify({"message": "Audio format should be mp3/wav", "status": 400}), 400

    # 使用临时目录保存上传文件
    temp_dir = tempfile.mkdtemp()
    try:
        video_path = os.path.join(temp_dir, video_file.filename)
        audio_path = os.path.join(temp_dir, audio_file.filename)
        video_file.save(video_path)
        audio_file.save(audio_path)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = f"{timestamp}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # 使用修改后的简化版脚本
        cmd = [
            "python3", "/workspace/wav2lip-onnx-256/inference_trt_gfpgan_simplified.py",
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path
        ]
        logging.info("Running: %s", " ".join(cmd))

        # 实时把子进程 stdout/stderr 打印到终端
        with subprocess.Popen(cmd,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              bufsize=1,
                              universal_newlines=True) as proc:
            for line in proc.stdout:
                print(line, end='')          # 实时打印进度
            proc.wait()
            if proc.returncode != 0:
                logging.error("Inference failed")
                return jsonify({"message": "Inference failed", "status": 500}), 500

        logging.info("Inference finished: %s", output_path)
        return jsonify({"status": "success", "url": f"/file/{output_filename}"}), 200

    finally:
        # 清理临时目录
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=120)