"""
레이저 자동 검출 및 학습 시스템
프로젝터 스크린에서 미세한 레이저 점을 자동으로 찾고 학습
"""
import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import os
import json
import pickle
from datetime import datetime
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import DBSCAN

@dataclass
class LaserPoint:
    """검출된 레이저 포인트 정보"""
    x: int
    y: int
    confidence: float
    brightness: float
    green_ratio: float
    contrast: float
    image_id: str

class LaserAutoLearner:
    """레이저 자동 검출 및 학습 시스템"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        self.on_dir = os.path.join(data_dir, "images_on")
        self.off_dir = os.path.join(data_dir, "images_off")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # 검출된 레이저 포인트들
        self.detected_points: List[LaserPoint] = []
        
        # 학습된 파라미터
        self.learned_params = {
            'brightness_range': (0, 255),
            'green_ratio_range': (0.0, 1.0),
            'contrast_range': (0, 255),
            'avg_position': (0, 0),
            'position_std': 0,
            'confidence_threshold': 0.5
        }
        
        # 통계 정보
        self.stats = {
            'total_pairs': 0,
            'detected_count': 0,
            'failed_count': 0,
            'avg_confidence': 0.0
        }
        
        os.makedirs(self.processed_dir, exist_ok=True)
    
    def load_image_pairs(self) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """ON/OFF 이미지 쌍 로드"""
        on_files = sorted([f for f in os.listdir(self.on_dir) if f.endswith(('.jpg', '.png'))])
        off_files = sorted([f for f in os.listdir(self.off_dir) if f.endswith(('.jpg', '.png'))])
        
        pairs = []
        for on_file in on_files:
            # 매칭되는 OFF 파일 찾기 (파일명 기준)
            base_name = on_file.replace('_on', '').replace('on_', '')
            matching_off = None
            
            for off_file in off_files:
                if base_name in off_file or off_file.replace('_off', '').replace('off_', '') == base_name:
                    matching_off = off_file
                    break
            
            if matching_off:
                on_path = os.path.join(self.on_dir, on_file)
                off_path = os.path.join(self.off_dir, matching_off)
                
                on_img = cv2.imread(on_path)
                off_img = cv2.imread(off_path)
                
                if on_img is not None and off_img is not None:
                    pairs.append((on_img, off_img, on_file))
                    print(f"✅ 로드: {on_file} <-> {matching_off}")
        
        print(f"\n총 {len(pairs)}개 이미지 쌍 로드 완료")
        return pairs
    
    def detect_laser_difference(self, on_img: np.ndarray, off_img: np.ndarray) -> Optional[LaserPoint]:
        """히트맵 기반 단순 검출"""
        diff = cv2.absdiff(on_img, off_img)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray_diff, (15, 15), 0)
        
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
        
        if max_val < 10:  # 최소 임계값
            return None
            
        x, y = max_loc
        
        # 간단한 특징 추출 (기존 방식 재사용)
        features = self._extract_features(on_img, off_img, x, y)
        if not features:
            return None
            
        return LaserPoint(
            x=x, y=y,
            confidence=features['confidence'],
            brightness=features['brightness'], 
            green_ratio=features['green_ratio'],
            contrast=features['contrast'],
            image_id=""
        )
    
    def _extract_features(self, on_img: np.ndarray, off_img: np.ndarray, 
                         x: int, y: int) -> Optional[Dict]:
        """레이저 위치에서 특징 추출"""
        try:
            # ROI 추출 (5x5 영역)
            roi_size = 2
            x1, y1 = max(0, x-roi_size), max(0, y-roi_size)
            x2, y2 = min(on_img.shape[1], x+roi_size+1), min(on_img.shape[0], y+roi_size+1)
            
            on_roi = on_img[y1:y2, x1:x2]
            off_roi = off_img[y1:y2, x1:x2]
            
            # ON 이미지에서의 최대 밝기
            on_gray = cv2.cvtColor(on_roi, cv2.COLOR_BGR2GRAY)
            on_brightness = np.max(on_gray)
            
            # OFF 이미지에서의 평균 밝기
            off_gray = cv2.cvtColor(off_roi, cv2.COLOR_BGR2GRAY)
            off_brightness = np.mean(off_gray)
            
            # 대비 (차이)
            contrast = on_brightness - off_brightness
            
            # 중심 픽셀의 색상 정보
            center_bgr = on_img[y, x]
            b, g, r = float(center_bgr[0]), float(center_bgr[1]), float(center_bgr[2])
            
            # 녹색 비율
            total = b + g + r
            green_ratio = g / total if total > 0 else 0
            
            # 신뢰도 계산
            confidence = self._calculate_confidence(contrast, green_ratio, on_brightness)
            
            return {
                'brightness': on_brightness,
                'green_ratio': green_ratio,
                'contrast': contrast,
                'confidence': confidence
            }
            
        except Exception as e:
            print(f"특징 추출 오류: {e}")
            return None
    
    def _calculate_confidence(self, contrast: float, green_ratio: float, brightness: float) -> float:
        """신뢰도 계산"""
        # 대비 점수 (0~1)
        contrast_score = min(1.0, contrast / 50.0) if contrast > 0 else 0
        
        # 녹색 비율 점수 (0~1)
        green_score = min(1.0, max(0, (green_ratio - 0.3) / 0.3))
        
        # 밝기 점수 (0~1)
        brightness_score = min(1.0, brightness / 200.0) if brightness > 50 else 0
        
        # 가중 평균
        confidence = (contrast_score * 0.5 + green_score * 0.3 + brightness_score * 0.2)
        
        return confidence
    
    def process_all_pairs(self):
        """모든 이미지 쌍 처리"""
        print("\n=== 레이저 자동 검출 시작 ===")
        
        # 이미지 쌍 로드
        pairs = self.load_image_pairs()
        self.stats['total_pairs'] = len(pairs)
        
        # 각 쌍 처리
        for on_img, off_img, img_id in tqdm(pairs, desc="처리중"):
            laser_point = self.detect_laser_difference(on_img, off_img)
            
            if laser_point:
                laser_point.image_id = img_id
                self.detected_points.append(laser_point)
                self.stats['detected_count'] += 1
                
                # 시각화 저장 (옵션)
                self._save_detection_result(on_img, off_img, laser_point)
            else:
                self.stats['failed_count'] += 1
                print(f"⚠️ 검출 실패: {img_id}")
        
        print(f"\n검출 완료: {self.stats['detected_count']}/{self.stats['total_pairs']}")
    
    def _save_detection_result(self, on_img: np.ndarray, off_img: np.ndarray, point: LaserPoint):
        """검출 결과 시각화 저장"""
        # 차이 이미지
        diff = cv2.absdiff(on_img, off_img)
        
        # 마킹
        marked_img = on_img.copy()
        cv2.circle(marked_img, (point.x, point.y), 10, (0, 255, 0), 2)
        cv2.circle(marked_img, (point.x, point.y), 1, (0, 0, 255), -1)
        
        # 정보 텍스트
        text = f"Conf: {point.confidence:.2f}"
        cv2.putText(marked_img, text, (point.x + 15, point.y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 결합 이미지
        combined = np.hstack([off_img, on_img, diff, marked_img])
        
        # 저장
        output_path = os.path.join(self.processed_dir, f"detected_{point.image_id}")
        cv2.imwrite(output_path, combined)
    
    def analyze_and_learn(self):
        """검출된 포인트들 분석 및 학습"""
        if len(self.detected_points) < 10:
            print("⚠️ 검출된 포인트가 너무 적습니다 (최소 10개 필요)")
            return False
        
        print("\n=== 학습 시작 ===")
        
        # 1. 데이터 추출
        positions = [(p.x, p.y) for p in self.detected_points]
        brightnesses = [p.brightness for p in self.detected_points]
        green_ratios = [p.green_ratio for p in self.detected_points]
        contrasts = [p.contrast for p in self.detected_points]
        confidences = [p.confidence for p in self.detected_points]
        
        # 2. 이상치 제거 로직 (수정된 부분)
        # 모든 검출 포인트를 유효한(valid) 데이터로 간주합니다.
        print("✅ 모든 검출 포인트를 학습에 사용합니다 (이상치 제거 비활성화).")
        valid_positions = np.array(positions)
        valid_brightnesses = brightnesses
        valid_green_ratios = green_ratios
        valid_contrasts = contrasts
        
        # 3. 통계 계산
        self.learned_params['avg_position'] = tuple(np.mean(valid_positions, axis=0).astype(int))
        self.learned_params['position_std'] = np.std(valid_positions)
        
        # 4. 범위 계산 (평균 ± 2σ)
        bright_mean, bright_std = np.mean(valid_brightnesses), np.std(valid_brightnesses)
        self.learned_params['brightness_range'] = (
            max(0, bright_mean - 2*bright_std),
            min(255, bright_mean + 2*bright_std)
        )
        
        green_mean, green_std = np.mean(valid_green_ratios), np.std(valid_green_ratios)
        self.learned_params['green_ratio_range'] = (
            max(0, green_mean - 2*green_std),
            min(1.0, green_mean + 2*green_std)
        )
        
        contrast_mean, contrast_std = np.mean(valid_contrasts), np.std(valid_contrasts)
        self.learned_params['contrast_range'] = (
            max(0, contrast_mean - 2*contrast_std),
            contrast_mean + 2*contrast_std
        )
        
        # 5. 신뢰도 임계값
        self.learned_params['confidence_threshold'] = np.percentile(confidences, 30)
        
        # 6. 통계 업데이트
        self.stats['avg_confidence'] = np.mean(confidences)
        
        print("\n📊 학습된 파라미터:")
        print(f"  평균 위치: {self.learned_params['avg_position']}")
        print(f"  위치 표준편차: {self.learned_params['position_std']:.1f} pixels")
        print(f"  밝기 범위: {self.learned_params['brightness_range'][0]:.0f} ~ {self.learned_params['brightness_range'][1]:.0f}")
        print(f"  녹색비율 범위: {self.learned_params['green_ratio_range'][0]:.2f} ~ {self.learned_params['green_ratio_range'][1]:.2f}")
        print(f"  대비 범위: {self.learned_params['contrast_range'][0]:.0f} ~ {self.learned_params['contrast_range'][1]:.0f}")
        print(f"  신뢰도 임계값: {self.learned_params['confidence_threshold']:.2f}")
        
        return True
    
    def visualize_results(self):
        """학습 결과 시각화 (영문 버전)"""
        if len(self.detected_points) == 0:
            print("No data to visualize.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Laser Detection Learning Analysis', fontsize=16)
        
        # 1. 위치 분포 (Location Distribution)
        positions = [(p.x, p.y) for p in self.detected_points]
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        axes[0, 0].scatter(x_coords, y_coords, alpha=0.5)
        axes[0, 0].set_title('Laser Position Distribution')
        axes[0, 0].set_xlabel('X Coordinate')
        axes[0, 0].set_ylabel('Y Coordinate')
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)
        
        # 2. 밝기 히스토그램 (Brightness Histogram)
        brightnesses = [p.brightness for p in self.detected_points]
        axes[0, 1].hist(brightnesses, bins=25, edgecolor='black')
        axes[0, 1].set_title('Brightness Distribution')
        axes[0, 1].set_xlabel('Brightness Value')
        axes[0, 1].axvline(self.learned_params['brightness_range'][0], color='r', linestyle='--', label=f"Min: {self.learned_params['brightness_range'][0]:.0f}")
        axes[0, 1].axvline(self.learned_params['brightness_range'][1], color='r', linestyle='--', label=f"Max: {self.learned_params['brightness_range'][1]:.0f}")
        axes[0, 1].legend()
        
        # 3. 녹색 비율 (Green Ratio)
        green_ratios = [p.green_ratio for p in self.detected_points]
        axes[0, 2].hist(green_ratios, bins=25, edgecolor='black', color='green')
        axes[0, 2].set_title('Green Ratio Distribution')
        axes[0, 2].set_xlabel('Green Channel Ratio')
        
        # 4. 대비 (Contrast)
        contrasts = [p.contrast for p in self.detected_points]
        axes[1, 0].hist(contrasts, bins=25, edgecolor='black', color='orange')
        axes[1, 0].set_title('Contrast Distribution')
        axes[1, 0].set_xlabel('Contrast (ON - OFF Brightness)')
        
        # 5. 신뢰도 (Confidence)
        confidences = [p.confidence for p in self.detected_points]
        axes[1, 1].hist(confidences, bins=25, edgecolor='black', color='purple')
        axes[1, 1].set_title('Confidence Score Distribution')
        axes[1, 1].set_xlabel('Confidence')
        axes[1, 1].axvline(self.learned_params['confidence_threshold'], color='r', linestyle='--', label=f"Threshold: {self.learned_params['confidence_threshold']:.2f}")
        axes[1, 1].legend()
        
        # 6. 시간별 신뢰도 (Confidence over Time)
        axes[1, 2].plot(confidences, 'o-', alpha=0.5)
        axes[1, 2].set_title('Confidence per Detection Order')
        axes[1, 2].set_xlabel('Image Index')
        axes[1, 2].set_ylabel('Confidence Score')
        axes[1, 2].axhline(self.learned_params['confidence_threshold'], color='r', linestyle='--', label=f"Threshold")
        axes[1, 2].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
        self.fig = fig
        # plt.savefig('results/learning_analysis.png', dpi=150) # process_experiments.py에서 저장하므로 여기선 필요 없음
        # plt.show()
    
    def save_model(self, model_path: str = "models/laser_model.pkl"):
        """학습된 모델 저장"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'params': self.learned_params,
            'stats': self.stats,
            'detected_points': [asdict(p) for p in self.detected_points],
            'timestamp': datetime.now().isoformat()
        }
        
        # Pickle 저장
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # JSON 백업 (사람이 읽을 수 있음)
        json_path = model_path.replace('.pkl', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ 모델 저장 완료: {model_path}")
    
    def load_model(self, model_path: str = "models/laser_model.pkl") -> bool:
        """저장된 모델 로드"""
        if not os.path.exists(model_path):
            print(f"❌ 모델 파일이 없습니다: {model_path}")
            return False
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.learned_params = model_data['params']
            self.stats = model_data['stats']
            self.detected_points = [LaserPoint(**p) for p in model_data['detected_points']]
            
            print(f"✅ 모델 로드 완료")
            print(f"  학습 시간: {model_data['timestamp']}")
            print(f"  검출 포인트: {len(self.detected_points)}개")
            
            return True
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            return False
    
    def test_on_new_image(self, on_img: np.ndarray, off_img: np.ndarray) -> Optional[Tuple[int, int]]:
        """학습된 파라미터로 새 이미지 테스트"""
        point = self.detect_laser_difference(on_img, off_img)
        
        if point and point.confidence >= self.learned_params['confidence_threshold']:
            # 학습된 범위 내에 있는지 확인
            if (self.learned_params['brightness_range'][0] <= point.brightness <= self.learned_params['brightness_range'][1] and
                self.learned_params['green_ratio_range'][0] <= point.green_ratio <= self.learned_params['green_ratio_range'][1]):
                return (point.x, point.y)
        
        return None


# 메인 실행 코드
if __name__ == "__main__":
    print("=== 레이저 자동 학습 시스템 ===")
    
    # 1. 학습기 초기화
    learner = LaserAutoLearner(data_dir="./data")
    
    # 2. 모든 이미지 쌍 처리
    learner.process_all_pairs()
    
    # 3. 분석 및 학습
    if learner.analyze_and_learn():
        # 4. 결과 시각화
        learner.visualize_results()
        
        # 5. 모델 저장
        learner.save_model()
        
        print("\n✅ 학습 완료!")
        print(f"총 {learner.stats['detected_count']}개 포인트에서 학습")
        print(f"평균 신뢰도: {learner.stats['avg_confidence']:.2f}")
    else:
        print("\n❌ 학습 실패 - 데이터 부족")