#!/bin/bash
# 레이저 검출 학습 환경 설정 스크립트

# 1. 가상환경 생성
echo "=== 가상환경 생성 ==="
python -m venv laser_env

# 2. 가상환경 활성화 (Windows)
# laser_env\Scripts\activate

# 2. 가상환경 활성화 (Linux/Mac)
source laser_env/bin/activate

# 3. 필요 패키지 설치
echo "=== 패키지 설치 ==="
pip install --upgrade pip
pip install opencv-python==4.8.1.78
pip install numpy==1.24.3
pip install matplotlib==3.7.2
pip install scikit-learn==1.3.0
pip install scipy==1.11.1
pip install Pillow==10.0.0
pip install tqdm==4.66.1

# 4. 프로젝트 디렉토리 구조 생성
echo "=== 디렉토리 구조 생성 ==="
mkdir -p laser_detection_project
cd laser_detection_project

mkdir -p data/images_on
mkdir -p data/images_off
mkdir -p data/processed
mkdir -p models
mkdir -p results
mkdir -p logs

echo "=== 설치 완료 ==="
echo "이제 이미지를 다음 위치에 넣어주세요:"
echo "  - 레이저 ON: data/images_on/"
echo "  - 레이저 OFF: data/images_off/"