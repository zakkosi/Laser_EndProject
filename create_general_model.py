import os
import cv2
import glob
from tqdm import tqdm
from laser_learning import LaserAutoLearner

# --- 설정 ---
ROOT_DIR = "raw_images"

def create_general_model():
    """
    모든 실험 환경의 데이터를 종합하여 하나의 통합 모델을 생성합니다.
    """
    print(f"\n{'='*20}")
    print("🚀 통합 모델 생성을 시작합니다.")
    print(f"{'='*20}")

    # 모든 환경에서 검출된 레이저 포인트를 저장할 리스트
    all_detected_points = []
    total_images_processed = 0

    # 1. 모든 실험 폴더를 순회하며 데이터 수집
    experiment_folders = [f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))]
    
    for exp_folder in experiment_folders:
        exp_path = os.path.join(ROOT_DIR, exp_folder)
        print(f"\n📂 데이터 수집 중: [{exp_folder}]")
        
        base_files = glob.glob(os.path.join(exp_path, "*_base.*"))
        if not base_files: continue
        base_image_path = base_files[0]
        
        data_folder_path = os.path.join(exp_path, "data")
        if not os.path.isdir(data_folder_path): continue
        data_image_paths = glob.glob(os.path.join(data_folder_path, "*.*"))
        if not data_image_paths: continue

        # 임시 학습기 객체 (검출 기능만 사용)
        temp_learner = LaserAutoLearner()
        off_img = cv2.imread(base_image_path)
        if off_img is None: continue

        for on_img_path in tqdm(data_image_paths, desc=f"[{exp_folder}] 검출 중"):
            on_img = cv2.imread(on_img_path)
            if on_img is None: continue
            
            laser_point = temp_learner.detect_laser_difference(on_img, off_img)
            if laser_point:
                laser_point.image_id = f"{exp_folder}/{os.path.basename(on_img_path)}"
                all_detected_points.append(laser_point)
        
        total_images_processed += len(data_image_paths)

    print(f"\n📊 총 {len(all_detected_points)} / {total_images_processed} 개의 레이저 포인트를 수집했습니다.")

    # 2. 수집된 모든 데이터로 단일 모델 학습
    if len(all_detected_points) < 20: # 통합 모델은 더 많은 데이터가 필요
        print("❌ 데이터가 너무 적어 통합 모델을 만들 수 없습니다.")
        return

    general_learner = LaserAutoLearner(data_dir="./data/GENERAL")
    general_learner.detected_points = all_detected_points # 수집한 모든 데이터를 주입

    if general_learner.analyze_and_learn():
        general_learner.visualize_results()
        
        plt_path = "results/learning_analysis_GENERAL.png"
        general_learner.fig.savefig(plt_path, dpi=150)
        print(f"📈 통합 분석 그래프 저장 완료: {plt_path}")

        model_path = "models/laser_model_GENERAL.pkl"
        general_learner.save_model(model_path)
        print(f"\n✅ 통합 모델 생성 완료!")
    else:
        print("\n❌ 통합 모델 학습 실패.")


if __name__ == "__main__":
    create_general_model()