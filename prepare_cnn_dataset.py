import os
import cv2
import glob
from tqdm import tqdm
from laser_learning import LaserAutoLearner

ROOT_DIR = "raw_images"

def create_dataset_with_diff_images():
    """
    모든 실험 환경에서 차집합 이미지와 좌표를 수집하여 데이터셋 생성
    """
    print("차집합 이미지 기반 데이터셋 생성 시작")
    
    # 데이터셋 저장 디렉토리
    os.makedirs("dataset/diff_images", exist_ok=True)
    os.makedirs("dataset/coordinates", exist_ok=True)
    
    all_data = []
    dataset_index = 0
    
    experiment_folders = [f for f in os.listdir(ROOT_DIR) if os.path.isdir(os.path.join(ROOT_DIR, f))]
    
    for exp_folder in experiment_folders:
        exp_path = os.path.join(ROOT_DIR, exp_folder)
        print(f"처리 중: [{exp_folder}]")
        
        # base 이미지와 data 폴더 찾기
        base_files = glob.glob(os.path.join(exp_path, "*_base.*"))
        if not base_files: continue
        
        data_folder_path = os.path.join(exp_path, "data")
        if not os.path.isdir(data_folder_path): continue
        data_image_paths = glob.glob(os.path.join(data_folder_path, "*.*"))
        if not data_image_paths: continue

        # OFF 이미지 로드
        off_img = cv2.imread(base_files[0])
        if off_img is None: continue
        
        temp_learner = LaserAutoLearner()
        img_height = off_img.shape[0]

        for on_img_path in tqdm(data_image_paths, desc=f"[{exp_folder}] 생성 중"):
            on_img = cv2.imread(on_img_path)
            if on_img is None: continue
            
            # 레이저 검출
            laser_point = temp_learner.detect_laser_difference(on_img, off_img)
            if laser_point and laser_point.y < img_height / 2:  # Y축 필터링
                
                # 차집합 이미지 저장
                diff = cv2.absdiff(on_img, off_img)
                gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                
                diff_path = f"dataset/diff_images/{dataset_index:05d}.png"
                cv2.imwrite(diff_path, gray_diff)
                
                # 좌표와 메타데이터 저장
                data_entry = {
                    'diff_image_path': diff_path,
                    'x': laser_point.x,
                    'y': laser_point.y,
                    'environment': exp_folder,
                    'original_file': os.path.basename(on_img_path)
                }
                all_data.append(data_entry)
                dataset_index += 1
    
    # 전체 데이터셋 정보 저장
    import json
    with open("dataset/dataset_info.json", 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"데이터셋 생성 완료: {len(all_data)}개 샘플")
    return all_data

if __name__ == "__main__":
    create_dataset_with_diff_images()