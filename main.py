"""
레이저 학습 실행 스크립트
이미지 준비부터 학습, 테스트까지 전체 파이프라인
"""

import os
import sys
from pathlib import Path

# 위에서 만든 모듈 임포트
from laser_learning import LaserAutoLearner

def setup_directories(base_dir="."):
    """디렉토리 구조 생성"""
    dirs = [
        f"{base_dir}/raw_images", # 원본 이미지 저장 위치
        f"{base_dir}/data/images_on",
        f"{base_dir}/data/images_off", 
        f"{base_dir}/data/processed",
        f"{base_dir}/models",
        f"{base_dir}/results",
        f"{base_dir}/logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ 디렉토리 구조 생성 완료")
    print("➡️ 원본 이미지를 'raw_images' 폴더에 넣어주세요.")

def organize_images(source_dir: str, on_dir: str, off_dir: str):
    """
    이미지를 ON/OFF 폴더로 자동 분류
    파일명에 'on' 또는 'off'가 포함되어 있어야 함
    """
    if not os.path.exists(source_dir):
        print(f"❌ 소스 디렉토리가 없습니다: {source_dir}")
        return False
    
    on_count = 0
    off_count = 0
    
    print(f"\n📂 '{source_dir}' 폴더에서 이미지 분류를 시작합니다...")
    for file in os.listdir(source_dir):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        
        src_path = os.path.join(source_dir, file)
        
        # 파일명으로 ON/OFF 구분
        if 'on' in file.lower():
            dst_path = os.path.join(on_dir, file)
            os.rename(src_path, dst_path)
            on_count += 1
        elif 'off' in file.lower():
            dst_path = os.path.join(off_dir, file)
            os.rename(src_path, dst_path)
            off_count += 1
    
    print(f"✅ 이미지 분류 완료: ON={on_count}, OFF={off_count}")
    return True

def validate_image_pairs(on_dir: str, off_dir: str):
    """이미지 쌍 유효성 검증"""
    on_files = set(os.listdir(on_dir))
    off_files = set(os.listdir(off_dir))
    
    print(f"\n📊 이미지 쌍 유효성을 검증합니다:")
    print(f"  - ON 이미지: {len(on_files)}개")
    print(f"  - OFF 이미지: {len(off_files)}개")
    
    if not on_files or not off_files:
        print("❌ 한 종류 이상의 이미지가 없습니다! ON/OFF 폴더를 확인해주세요.")
        return False

    missing_pairs = []
    # ON 이미지를 기준으로 짝이 되는 OFF 이미지가 있는지 확인
    for on_file in on_files:
        # 파일 이름 규칙을 유연하게 처리 (예: cam1_on_001.png -> cam1_off_001.png)
        expected_off = on_file.replace('_on', '_off').replace(' on ', ' off ')
        if expected_off == on_file: # 이름에 'on'이 없어 변경이 안된 경우
             print(f"⚠️ 경고: ON 파일 '{on_file}'의 이름에 'on' 키워드가 없어 짝을 찾기 어렵습니다.")
             continue
        
        if expected_off not in off_files:
            missing_pairs.append((on_file, expected_off))

    if not missing_pairs:
        print("✅ 모든 ON/OFF 이미지가 성공적으로 쌍을 이룹니다.")
        return True
    else:
        print("❌ 일부 이미지의 짝을 찾을 수 없습니다:")
        for on_f, off_f in missing_pairs:
            print(f"  - ON: {on_f}  ->  OFF: {off_f} (찾을 수 없음)")
        return False


if __name__ == "__main__":
    
    # --- 1. 기본 디렉토리 설정 ---
    setup_directories()
    
    # --- 2. 원본 이미지 자동 분류 ---
    # 'raw_images' 폴더에 있는 이미지를 'data/images_on', 'data/images_off'로 이동
    organize_images("raw_images", "data/images_on", "data.images_off")
    
    # --- 3. 이미지 쌍 유효성 검증 ---
    if not validate_image_pairs("data/images_on", "data/images_off"):
        print("\n🛑 학습을 진행할 수 없습니다. 이미지 파일 쌍을 확인해주세요.")
        sys.exit() # 프로그램 종료
        
    print("\n🚀 모든 준비가 완료되었습니다. 레이저 검출 학습을 시작합니다.")
    
    # --- 4. 레이저 학습 실행 ---
    learner = LaserAutoLearner(data_dir="./data")
    
    # 모든 ON/OFF 쌍 이미지에서 레이저 위치 검출
    learner.process_all_pairs()
    
    # 검출된 포인트들을 분석하여 최적의 파라미터 학습
    if learner.analyze_and_learn():
        # 학습 결과 시각화 및 저장
        learner.visualize_results()
        
        # 학습된 파라미터 모델로 저장
        learner.save_model()
        
        print("\n🎉 최종 학습 완료!")
        print(f"  - 총 {learner.stats['detected_count']}개의 유효한 레이저 포인트로 학습했습니다.")
        print(f"  - 평균 신뢰도: {learner.stats['avg_confidence']:.2f}")
        print("  - 학습 결과는 'results' 폴더와 'models' 폴더를 확인하세요.")
    else:
        print("\n❌ 학습 실패: 검출된 레이저 포인트가 너무 적습니다.")