import os
import cv2
import glob
from tqdm import tqdm
from laser_learning import LaserAutoLearner

# --- 설정 ---
# 실험 폴더들이 있는 루트 디렉토리
ROOT_DIR = "raw_images"

def run_experiment(exp_name: str, base_image_path: str, data_image_paths: list):
    """
    하나의 실험 단위(예: Dark_Black_background)에 대한 학습을 수행합니다.
    
    Args:
        exp_name (str): 실험 이름 (폴더명)
        base_image_path (str): 기준(OFF) 이미지 파일 경로
        data_image_paths (list): 레이저(ON) 이미지 파일 경로 리스트
    """
    print(f"\n{'='*20}")
    print(f"🚀 실험 시작: {exp_name}")
    print(f"🔬 기준(OFF) 이미지: {os.path.basename(base_image_path)}")
    print(f"🔢 레이저(ON) 이미지 수: {len(data_image_paths)}개")
    print(f"{'='*20}")
    
    # 1. 학습기 초기화
    # 결과물이 덮어쓰이지 않도록 processed 폴더를 실험별로 지정
    learner = LaserAutoLearner(data_dir=f"./data/{exp_name}")

    # 2. 기준(OFF) 이미지 로드 (한 번만 로드)
    off_img = cv2.imread(base_image_path)
    if off_img is None:
        print(f"❌ 기준 이미지 로드 실패: {base_image_path}")
        return

    # 3. 모든 ON 이미지와 페어링하여 레이저 검출
    for on_img_path in tqdm(data_image_paths, desc=f"[{exp_name}] 처리중"):
        on_img = cv2.imread(on_img_path)
        if on_img is None:
            continue

        laser_point = learner.detect_laser_difference(on_img, off_img)
        
        if laser_point:
            laser_point.image_id = os.path.basename(on_img_path)
            learner.detected_points.append(laser_point)
            learner.stats['detected_count'] += 1
            learner._save_detection_result(on_img, off_img, laser_point)
        else:
            learner.stats['failed_count'] += 1
    
    learner.stats['total_pairs'] = len(data_image_paths)
    print(f"\n검출 완료: {learner.stats['detected_count']}/{learner.stats['total_pairs']}")

    # 4. 분석 및 학습
    if learner.analyze_and_learn():
        # 5. 결과 시각화 및 모델 저장 (파일 이름에 실험명 추가)
        learner.visualize_results() # 화면에 표시
        plt_path = f"results/learning_analysis_{exp_name}.png"
        learner.fig.savefig(plt_path, dpi=150) # 수정된 부분: 그림 객체를 저장
        print(f"📈 분석 그래프 저장 완료: {plt_path}")

        model_path = f"models/laser_model_{exp_name}.pkl"
        learner.save_model(model_path)
        print(f"\n✅ [{exp_name}] 실험 학습 완료!")
    else:
        print(f"\n❌ [{exp_name}] 실험 학습 실패 - 데이터 부족")


if __name__ == "__main__":
    # raw_images 하위의 모든 폴더를 개별 실험으로 간주
    experiment_folders = [f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))]

    if not experiment_folders:
        print(f"❌ '{ROOT_DIR}' 폴더에 실험 폴더가 없습니다.")
    
    for exp_folder in experiment_folders:
        exp_path = os.path.join(ROOT_DIR, exp_folder)
        
        # 1. 기준(base) 이미지 찾기
        # glob을 사용해 이름이 _base로 끝나는 이미지 파일을 찾음
        base_files = glob.glob(os.path.join(exp_path, "*_base.*"))
        if not base_files:
            print(f"⚠️ [{exp_folder}] 폴더에 기준(*_base.*) 이미지가 없어 건너뜁니다.")
            continue
        base_image_path = base_files[0]
        
        # 2. 데이터(ON) 이미지 리스트 찾기
        data_folder_path = os.path.join(exp_path, "data")
        if not os.path.isdir(data_folder_path):
            print(f"⚠️ [{exp_folder}] 폴더에 'data' 하위 폴더가 없어 건너뜁니다.")
            continue
        
        data_image_paths = glob.glob(os.path.join(data_folder_path, "*.*"))
        if not data_image_paths:
            print(f"⚠️ [{exp_folder}/data] 폴더에 레이저 이미지가 없어 건너뜁니다.")
            continue
            
        # 3. 해당 환경에 대한 실험 실행
        run_experiment(exp_folder, base_image_path, data_image_paths)

    print("\n\n🎉 모든 실험이 종료되었습니다.")