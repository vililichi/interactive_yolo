#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd "$SCRIPT_DIR"

python3 -m venv ./venv --system-site-packages --symlinks
source ./venv/bin/activate
touch ./venv/COLCON_IGNORE

pip install "numpy<2.0" opencv-python-headless matplotlib torch torchvision "ultralytics<8.3.112" pyqt5 torchmetrics[detection] cv-bridge lz4
pip install pyaudio sounddevice SpeechRecognition vosk yake playsound gtts
pip install -e src/common/python/
pip install git+https://github.com/openai/CLIP.git

