"""
Virtra RGB 레이저 검출기 핵심 모듈 (RGB Core Detection Module)
HSV 기반 검출을 완전 제거하고 RGB + 프레임 차이 + 하이브리드 검출로 재구성

핵심 기능:
- RGB 밝기 기반 레이저 검출 (detect_laser_candidates)
- 프레임 차이 기반 움직임 검출 (motion detection)
- 하이브리드 검출 알고리즘 (RGB + Motion + Depth)
- 적응형 임계값 학습 시스템 (adaptive thresholding)
- 검출 결과 처리 및 검증
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

# 🔬 과학적 레이저 검출을 위한 추가 임포트
try:
    from scientific_laser_scoring_system import ScientificLaserScoringSystem
    SCIENTIFIC_SCORING_AVAILABLE = True
    print("[INFO] 과학적 레이저 스코어링 시스템 로드됨")
except ImportError:
    SCIENTIFIC_SCORING_AVAILABLE = False
    print("[INFO] 과학적 스코어링 시스템 없음 - 기본 모드 사용")


@dataclass
class LaserDetectionResult:
    """RGB 레이저 검출 결과"""
    detected: bool
    confidence: float
    position: Tuple[int, int]
    rgb_values: Tuple[int, int, int]  # RGB 값 (r, g, b)
    brightness: float
    detection_method: str
    detection_time_ms: float
    # 2D 스크린 매핑 결과 (다른 모듈에서 설정)
    screen_coordinate: Optional[Tuple[float, float]] = None
    unity_coordinate: Optional[Tuple[float, float]] = None
    # 3D 정보 (깊이 센서 사용시)
    world_3d_point: Optional[Tuple[float, float, float]] = None  # 3D 월드 좌표
    depth_mm: Optional[float] = None  # 깊이 값 (mm)
    # 움직임 정보
    motion_detected: bool = False
    motion_intensity: float = 0.0


class LaserDetectorCore:
    """
    RGB 레이저 검출기 핵심 모듈
    
    담당 기능:
    - RGB 밝기 기반 레이저 검출
    - 프레임 차이 기반 움직임 검출
    - 하이브리드 검출 알고리즘 (RGB + Motion + Depth)
    - 적응형 임계값 학습
    - 검출 결과 처리 및 검증
    """
    
    def __init__(self):
        """RGB 검출 모듈 초기화"""
        # RGB 기반 검출 파라미터 (엄격한 임계값으로 거짓 양성 제거)
        self.brightness_threshold = 120   # 기본 밝기 임계값 (거짓 양성 제거: 60→120)
        self.adaptive_brightness_threshold = 120  # 적응형 밝기 임계값 (거짓 양성 제거: 60→120)
        self.min_laser_area = 5  # 최소 레이저 면적 (픽셀) - 점 노이즈 억제 강화 (3→5)
        self.max_laser_area = 500  # 최대 레이저 면적 (픽셀)
        
        # 과학적 스코어/학습/HSV 경로 비활성화(간단 파이프라인 고정)
        self.scientific_scorer = None
        
        # 🎯 프레임 차이 기반 움직임 검출 (레이저 최적화)
        self.motion_threshold = 35  # 움직임 검출 임계값 (노이즈 감소를 위해 상향 조정)
        self.previous_frame = None
        self.motion_mask = None
        self.motion_history = []  # 움직임 히스토리 (노이즈 필터링용)
        
        # 하이브리드 가중치/깊이 보너스 비사용
        self.brightness_weight = 0.0
        self.motion_weight = 0.0
        self.area_weight = 0.0
        self.depth_weight = 0.0
        
        # 🎯 레이저 vs 일반 움직임 구분을 위한 고급 필터링
        self.require_motion_for_detection = False
        self.min_confidence_threshold = 0.0
        self.min_confidence_threshold_screen = 0.0
        self.min_confidence_threshold_gun = 0.0
        self.consecutive_detection_required = 1    # 연속 검출 필요 횟수 (즉시 복원: 3→1)
        self.detection_history = []  # 최근 검출 이력
        
        # 고급 파라미터 비활성화
        self.laser_motion_area_threshold = 999999
        self.laser_brightness_threshold = 0
        self.motion_laser_confidence_threshold = 0.0
        self.scientific_score_threshold = 0.0
        
        # ROI (Region of Interest) 설정
        self.enable_roi = False  # ROI 사용 여부
        self.roi_bounds = None   # ROI 경계 (x1, y1, x2, y2)
        self.screen_roi = None   # 스크린 전용 ROI
        self.gun_roi = None      # 총구 전용 ROI
        self.screen_polygon: Optional[List[Tuple[int,int]]] = None  # 스크린 폴리곤 마스크(옵션)
        # 사용자 우선권(ROI 토글) 잠금. None=자동, True/False=사용자 고정
        self._roi_user_override: Optional[bool] = None
        
        # 스크린 캘리브레이션 기반 모드 (깊이 불필요)
        self.calibration_mode = True  # 캘리브레이션 기반 3D 변환
        self.screen_mode = True  # 스크린 평면 고정
        self.depth_enabled = False  # 깊이 센서 비활성화 (불필요)
        
        # 호환성을 위한 깊이 관련 속성 (사용하지 않지만 유지)
        self.depth_range_mm = (300, 5000)  # 호환성용
        
        # 학습 시스템
        self.learned_samples = []  # 학습된 샘플 데이터
        self.max_samples = 50
        
        # 검출 이력
        self.detection_history = []
        self.max_history = 20
        
        # 통계
        self.stats = {
            'total_detections': 0,
            'motion_detections': 0,
            'brightness_detections': 0,
            'hybrid_detections': 0
        }
        
        # 로깅
        self.logger = logging.getLogger(__name__)
        
        # 현재 프레임 상태
        self.current_frame = None
        self.current_gray = None
        
        # ----- Screen 전용 간소 파이프라인 파라미터 (물리 기반) -----
        # 녹색 우위/ExG/국소대비/DoG 점광 + 면적/원형도
        self.screen_ratio_threshold: float = 0.60   # g_ratio = G/(R+G+B)
        self.screen_exg_threshold: int = 60         # ExcessGreen = 2G - R - B
        self.screen_contrast_threshold: int = 40    # center - median7x7
        self.screen_log_threshold: int = 15         # DoG 반응 임계(0~255 정규화)
        self.screen_min_area: int = 4
        self.screen_max_area: int = 80
        self.screen_circularity_min: float = 0.35
        self.screen_min_confidence: float = 0.50    # 표시/채택 최소 신뢰도

        # 시간 안정화(N-of-M)
        self.screen_temporal_window: int = 5
        self.screen_temporal_needed: int = 3
        self.screen_temporal_radius: int = 3  # px
        self._screen_recent_positions: List[Tuple[int, int, float]] = []  # (x, y, t)

        print("[INFO] RGB Laser Detector Core Initialized (하이브리드 모드)")
    
    def update_motion_mask(self, current_frame: np.ndarray) -> np.ndarray:
        """프레임 차이 기반 움직임 마스크 생성"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        self.current_gray = gray
        
        if self.previous_frame is None:
            self.previous_frame = gray.copy()
            return np.zeros_like(gray)
        
        # 프레임 차이 계산
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        
        # 임계값 적용
        _, motion_mask = cv2.threshold(frame_diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
        
        # 노이즈 제거
        kernel = np.ones((3,3), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # 이전 프레임 업데이트
        self.previous_frame = gray.copy()
        self.motion_mask = motion_mask
        
        return motion_mask
    
    def set_roi(self, roi_type: str, bounds: Tuple[int, int, int, int]):
        """ROI 설정"""
        x1, y1, x2, y2 = bounds
        if roi_type == "screen":
            self.screen_roi = (x1, y1, x2, y2)
            # [quiet] 스크린 ROI 설정
        elif roi_type == "gun":
            self.gun_roi = (x1, y1, x2, y2)
            # [quiet] 총구 ROI 설정
        
        # ROI 활성화: 사용자가 명시적으로 끄지 않았다면 자동 활성화
        if self._roi_user_override is None:
            self.enable_roi = True
        else:
            self.enable_roi = bool(self._roi_user_override)

    def set_roi_enable(self, enabled: bool, user_override: bool = False):
        """ROI 전체 사용 여부 설정. user_override=True이면 이후 자동 갱신이 이 값을 존중"""
        if user_override:
            self._roi_user_override = bool(enabled)
        # 항상 사용자 오버라이드가 우선
        self.enable_roi = bool(self._roi_user_override) if self._roi_user_override is not None else bool(enabled)

    def set_screen_polygon(self, points: List[Tuple[int,int]]):
        """스크린 폴리곤 마스크 설정 (4점 이상, 시계/반시계 무관)
        - None 또는 빈 리스트를 전달하면 비활성화
        """
        try:
            if points and len(points) >= 3:
                self.screen_polygon = [(int(x), int(y)) for x,y in points]
                # [quiet] 스크린 폴리곤 설정
            else:
                self.screen_polygon = None
                # [quiet] 스크린 폴리곤 비활성화
        except Exception as e:
            self.logger.error(f"스크린 폴리곤 설정 실패: {e}")
    
    def _apply_roi_filter(self, image: np.ndarray, roi_type: str = "screen") -> np.ndarray:
        """ROI 마스크 적용"""
        if not self.enable_roi:
            return image
        
        # 스크린: 폴리곤이 있으면 폴리곤 우선, 없으면 사각 ROI
        if roi_type == "screen" and self.screen_polygon is not None and len(self.screen_polygon) >= 3:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            pts = np.array(self.screen_polygon, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        else:
            roi = self.screen_roi if roi_type == "screen" else self.gun_roi
            if roi is None:
                return image
            x1, y1, x2, y2 = roi
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            mask[y1:y2, x1:x2] = 255
        
        # ROI 외부 영역을 검은색으로 마스킹
        result = image.copy()
        result[mask == 0] = 0
        
        return result
    
    def detect_laser_candidates(self, image: np.ndarray, depth_frame: Optional[np.ndarray] = None, motion_mask: Optional[np.ndarray] = None, roi_type: str = "screen") -> List[Dict]:
        """
        RGB 하이브리드 레이저 검출 (HSV 완전 제거)
        
        검출 방식:
        1. 프레임 차이 기반 움직임 검출
        2. RGB 밝기 기반 검출  
        3. 깊이 정보 활용 (있는 경우)
        4. 하이브리드 신뢰도 계산
        
        Args:
            image: RGB 입력 이미지
            depth_frame: 깊이 프레임 (선택적)
            motion_mask: 외부 모션 마스크 (선택적)
            
        Returns:
            검출된 레이저 후보 목록
        """
        try:
            # 현재 프레임 저장
            self.current_frame = image.copy()
            candidates = []
            
            # ROI 필터 적용 (설정된 경우)
            filtered_image = self._apply_roi_filter(image, roi_type)
            
            # 1. 움직임 검출 (프레임 차이)
            if motion_mask is None:
                motion_mask = self.update_motion_mask(filtered_image)
            
            # 2. 후보 생성
            if roi_type == "screen":
                # 스크린 전용: 단순 물리 기반 파이프라인 사용
                brightness_candidates = self._detect_screen_simple_candidates(filtered_image, motion_mask)
            else:
                # 총구/기타: 기존 하이브리드 후보 생성
                brightness_candidates = self._detect_brightness_candidates(filtered_image, motion_mask, roi_type)
            
            # 3. 스크린 모드: 깊이 정보 불필요 (캘리브레이션 기반)
            candidates = brightness_candidates  # 깊이 처리 생략
            
            # 4. 간단 필터링: screen_simple은 자체 confidence 그대로 사용
            for c in candidates:
                c['confidence'] = float(min(1.0, max(0.0, c.get('confidence', 0.0))))
            
            # 신뢰도 순 정렬
            candidates.sort(key=lambda x: x['confidence'], reverse=True)
            
            # 통계 업데이트
            self.stats['total_detections'] += len(candidates)
            if any(c.get('motion_detected', False) for c in candidates):
                self.stats['motion_detections'] += 1
            
            return candidates[:10]  # 상위 10개 반환
            
        except Exception as e:
            self.logger.error(f"RGB 하이브리드 검출 오류: {e}")
            return []
    
    def _detect_brightness_candidates(self, image: np.ndarray, motion_mask: np.ndarray, roi_type: str = "screen") -> List[Dict]:
        """RGB 밝기 기반 레이저 후보 검출
        
        스크린 경로(screen): 밝기 마스크만으로 후보를 만들고, 모션은 신뢰도 가중치에만 사용
        총구 경로(gun): 밝기 마스크와 모션 마스크를 AND 결합하여 게이트 강화
        """
        candidates = []
        
        try:
            # 그레이스케일 변환
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 밝기 임계값 적용 (일원화: adaptive가 있으면 그 값, 없으면 brightness_threshold)
            # 스크린 경로 임계값 점진적 조정 (균형적 검출)
            if roi_type == "screen":
                effective_thr = int(max(80, min(150, self.adaptive_brightness_threshold)))  # 엄격화: 50-100 → 80-150
            else:
                effective_thr = int(max(50, min(255, self.adaptive_brightness_threshold)))  # 점진적 하향: 80 → 50
            # 키보드 조정 호환: 더 높은 값이 있으면 반영
            effective_thr = max(effective_thr, int(self.brightness_threshold))
            bright_mask = gray > effective_thr
            
            # 움직임 마스크와 결합 정책
            # - screen: 모션과 AND 결합하지 않음 (정지 레이저 허용)
            # - gun: 모션과 AND 결합하여 거짓 양성 감소
            if motion_mask is not None and roi_type != "screen":
                combined_mask = cv2.bitwise_and(bright_mask.astype(np.uint8) * 255, motion_mask)
            else:
                combined_mask = bright_mask.astype(np.uint8) * 255
            
            # 노이즈 제거
            kernel = np.ones((3,3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # 컨투어 검출
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 면적 필터링
                if self.min_laser_area <= area <= self.max_laser_area:
                    # 중심점 계산
                    M = cv2.moments(contour)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 경계 체크
                        if 0 <= cx < image.shape[1] and 0 <= cy < image.shape[0]:
                            # RGB 값 추출
                            b, g, r = image[cy, cx]
                            brightness = int(gray[cy, cx])
                            
                            # 🔬 과학적 레이저 검출 시스템 적용 (폴백 HSV 포함)
                            hsv_values = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
                            h, s, v = int(hsv_values[0]), int(hsv_values[1]), int(hsv_values[2])
                            scientific_scores = self._calculate_scientific_laser_score(h, s, v, r, g, b)
                            green_dominance = scientific_scores['final_confidence']
                            green_intensity = scientific_scores['bayesian_score']
                            
                            # 움직임 여부 확인
                            motion_detected = motion_mask[cy, cx] > 0 if motion_mask is not None else False
                            motion_intensity = float(motion_mask[cy, cx]) / 255.0 if motion_mask is not None else 0.0
                            
                            # 원형도 계산
                            perimeter = cv2.arcLength(contour, True)
                            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                            
                            # 원형도 필터 점진적 완화 (현실적 기준)
                            if circularity < 0.4:  # 엄격화: 0.3 → 0.4
                                continue
                            
                            # 스크린 경로 전용: 간단한 녹색 비율/HSV 게이트 선-필터
                            if roi_type == "screen":
                                total = int(r) + int(g) + int(b)
                                green_ratio = (g / total) if total > 0 else 0.0
                                # 엄격한 HSV 범위와 녹색 비율 기준 + 과학적 스코어 필터
                                hsv_ok = (50 <= h <= 75) and (s >= 70) and (v >= 80)  # 엄격화: H(45-85→50-75), S(50→70), V(50→80)
                                scientific_ok = green_dominance > 0.7  # 과학적 스코어 임계값 추가
                                if not (hsv_ok and green_ratio > 0.50 and scientific_ok):  # 엄격화: OR → AND, 과학적 스코어 추가
                                    continue

                            candidate = {
                                'position': (cx, cy),
                                'rgb_values': (int(r), int(g), int(b)),
                                'hsv_values': (h, s, v),  # HSV 값 추가
                                'brightness': brightness,
                                'area': area,
                                'circularity': circularity,
                                'motion_detected': motion_detected,
                                'motion_intensity': motion_intensity,
                                'detection_method': 'scientific_bayesian',  # 과학적 검출 표시
                                'contour': contour,
                                # 🔬 과학적 레이저 스코어링 정보
                                'scientific_scores': scientific_scores,
                                'green_dominance': green_dominance,
                                'green_intensity': green_intensity,
                                'bayesian_confidence': scientific_scores['bayesian_score'],
                                'physics_score': scientific_scores['physics_score'],
                                'is_green_laser': scientific_scores['is_laser_candidate']  # 과학적 판정
                            }
                            
                            candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"밝기 기반 검출 오류: {e}")
            return []

    def _detect_screen_simple_candidates(self, image: np.ndarray, motion_mask: Optional[np.ndarray]) -> List[Dict]:
        """스크린 전용 간소 파이프라인 후보 생성
        - 녹색비율 + Excess Green + 국소대비 + DoG 점광 + 면적/원형도
        """
        try:
            h, w = image.shape[:2]
            # 다운스케일에서 후보 찾기 → 원본에서 확정
            ds = 0.5
            small = cv2.resize(image, (int(w*ds), int(h*ds)))
            small_bgr = small
            small_gray = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2GRAY)

            # 색상 게이트: g_ratio & ExG
            b, g, r = cv2.split(small_bgr)
            total = cv2.add(cv2.add(r, g), b)
            total_safe = cv2.max(total, 1)
            g_ratio = (g.astype(np.float32) / total_safe.astype(np.float32))
            exg = (2*g.astype(np.int16) - r.astype(np.int16) - b.astype(np.int16)).clip(0, 255).astype(np.uint8)
            mask_color = (g_ratio > self.screen_ratio_threshold).astype(np.uint8) * 255
            _, mask_exg = cv2.threshold(exg, self.screen_exg_threshold, 255, cv2.THRESH_BINARY)
            mask_color = cv2.bitwise_and(mask_color, mask_exg)

            # 국소 대비: center - median7x7
            median7 = cv2.medianBlur(small_gray, 7)
            contrast = cv2.subtract(small_gray, median7)
            _, mask_contrast = cv2.threshold(contrast, self.screen_contrast_threshold, 255, cv2.THRESH_BINARY)

            # DoG 점광
            blur1 = cv2.GaussianBlur(small_gray, (0,0), 1.2)
            blur2 = cv2.GaussianBlur(small_gray, (0,0), 2.0)
            dog = cv2.subtract(blur1, blur2)
            dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)
            _, mask_dog = cv2.threshold(dog_norm, self.screen_log_threshold, 255, cv2.THRESH_BINARY)

            # 결합: 색상 ∧ (대비 ∨ DoG)로 완화/강화 조합
            contrast_or_dog = cv2.bitwise_or(mask_contrast, mask_dog)
            combined = cv2.bitwise_and(mask_color, contrast_or_dog)
            combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

            # 컨투어 분석(다운스케일 좌표)
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            candidates: List[Dict] = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < max(1, self.screen_min_area*ds*ds) or area > self.screen_max_area*ds*ds:
                    continue
                peri = cv2.arcLength(cnt, True)
                circularity = 4*np.pi*area/(peri*peri) if peri>0 else 0
                if circularity < self.screen_circularity_min:
                    continue
                M = cv2.moments(cnt)
                if M["m00"] <= 0:
                    continue
                cx_s = int(M["m10"]/M["m00"]) ; cy_s = int(M["m01"]/M["m00"]) 
                # 원본 좌표로 복원
                cx = int(cx_s/ds) ; cy = int(cy_s/ds)
                if not (0 <= cx < w and 0 <= cy < h):
                    continue

                B,G,R = image[cy, cx]
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                brightness = int(gray[cy, cx])
                total_pix = int(R)+int(G)+int(B)
                g_ratio_pix = (G/total_pix) if total_pix>0 else 0.0

                # gun 경로 하드 차단용 표식: screen_simple는 screen에서만 사용, gun에서는 무시될 예정

                candidates.append({
                    'position': (cx, cy),
                    'rgb_values': (int(R), int(G), int(B)),
                    'brightness': brightness,
                    'area': float(area/(ds*ds)),
                    'circularity': float(circularity),
                    'green_ratio': float(g_ratio_pix),
                    'detection_method': 'screen_simple'
                })

            # 신뢰도 간단 계산(색상/대비/형상 기반)
            for c in candidates:
                color_score = min(1.0, max(0.0, (c.get('green_ratio',0.0) - 0.58)/0.42))
                shape_score = min(1.0, max(0.0, (c.get('circularity',0.0) - 0.3)/0.7))
                brightness = c.get('brightness',0)
                contrast_score = min(1.0, max(0.0, (brightness - 160)/95))
                c['confidence'] = 0.55*color_score + 0.30*contrast_score + 0.15*shape_score

            # 시간 안정화: 최근 N 프레임 중 K 프레임 일치(±radius)
            now = time.time()
            stabilized: List[Dict] = []
            for c in candidates:
                cx, cy = c['position']
                count = 1
                # 과거 포지션들과 비교
                for px, py, ts in self._screen_recent_positions:
                    if now - ts > 0.6:  # 윈도우 축소로 잔광 억제
                        continue
                    if abs(px - cx) <= self.screen_temporal_radius and abs(py - cy) <= self.screen_temporal_radius:
                        count += 1
                if count >= max(2, self.screen_temporal_needed):
                    # 시간 보너스
                    c['confidence'] = min(1.0, c['confidence'] + 0.08)
                    stabilized.append(c)

            # 최근 포지션 버퍼 갱신
            if candidates:
                top = candidates[0]
                self._screen_recent_positions.append((top['position'][0], top['position'][1], now))
                # 윈도우 유지
                self._screen_recent_positions = [p for p in self._screen_recent_positions if now - p[2] <= 1.0][-self.screen_temporal_window:]

            # 최종 임계 적용: 화면 표시는 최소 신뢰도 이상만
            if stabilized:
                stabilized.sort(key=lambda x: x.get('confidence',0.0), reverse=True)
                stabilized = [s for s in stabilized if s.get('confidence',0.0) >= self.screen_min_confidence]
                return stabilized[:10]

            # 정렬 후 반환
            candidates.sort(key=lambda x: x.get('confidence',0.0), reverse=True)
            candidates = [c for c in candidates if c.get('confidence',0.0) >= self.screen_min_confidence]
            return candidates[:10]
        except Exception as e:
            self.logger.error(f"스크린 단순 후보 생성 오류: {e}")
            return []

    def detect_laser_points(self, image: np.ndarray) -> Optional[LaserDetectionResult]:
        """상위 후보 1개를 `LaserDetectionResult`로 변환하여 반환
        - 기존 호출부 호환을 위해 간단 래퍼를 제공한다.
        """
        try:
            start_time = time.time()
            candidates = self.detect_laser_candidates(image, depth_frame=None, motion_mask=None, roi_type="screen")
            if not candidates:
                return None
            top = candidates[0]
            cx, cy = map(int, top.get('position', (0, 0)))
            r, g, b = top.get('rgb_values', (0, 0, 0))
            brightness = int(top.get('brightness', 0))
            conf = float(top.get('confidence', 0.0))
            method = top.get('detection_method', 'rgb_hybrid')
            dt_ms = (time.time() - start_time) * 1000.0
            return LaserDetectionResult(
                detected=True,
                confidence=conf,
                position=(cx, cy),
                rgb_values=(r, g, b),
                brightness=brightness,
                detection_method=method,
                detection_time_ms=dt_ms
            )
        except Exception as e:
            self.logger.error(f"detect_laser_points 래퍼 오류: {e}")
            return None
    
    def _calculate_scientific_laser_score(self, h: int, s: int, v: int, r: int, g: int, b: int) -> Dict[str, float]:
        """
        🔬 과학적 레이저 스코어링 (베이지안 확률론 + 물리학적 모델)
        
        Args:
            h, s, v: HSV 값
            r, g, b: RGB 값
            
        Returns:
            과학적 검출 스코어 및 신뢰도
        """
        try:
            # 과학적 스코어링 시스템이 있으면 사용
            if self.scientific_scorer:
                scientific_result = self.scientific_scorer.calculate_scientific_score(h, s, v, 2000.0)
                
                # 레이저 후보 판정 (베이지안 + 물리학적 기준)
                bayesian_threshold = 0.6  # 베이지안 스코어 임계값
                physics_threshold = 0.5   # 물리학적 스코어 임계값
                
                is_laser_candidate = (
                    scientific_result['bayesian_score'] > bayesian_threshold or
                    scientific_result['physics_score'] > physics_threshold or
                    scientific_result['integrated_score'] > 0.55
                )
                
                return {
                    'bayesian_score': scientific_result['bayesian_score'],
                    'physics_score': scientific_result['physics_score'],
                    'integrated_score': scientific_result['integrated_score'],
                    'final_confidence': scientific_result['final_score'],
                    'confidence_lower': scientific_result['confidence_lower'],
                    'confidence_upper': scientific_result['confidence_upper'],
                    'is_laser_candidate': is_laser_candidate
                }
            
            # 폴백: 기본 HSV 기반 검출
            else:
                return self._fallback_hsv_detection(h, s, v, r, g, b)
                
        except Exception as e:
            self.logger.error(f"과학적 스코어링 오류: {e}")
            return self._fallback_hsv_detection(h, s, v, r, g, b)
    
    def _fallback_hsv_detection(self, h: int, s: int, v: int, r: int, g: int, b: int) -> Dict[str, float]:
        """폴백: 기본 HSV 검출 방식"""
        # 녹색 레이저 HSV 범위 (532nm 기준)
        green_h_range = (60, 80)  # 기본 녹색 범위
        min_saturation = 50       # 최소 채도
        min_value = 30           # 최소 명도
        
        # HSV 기반 점수 계산
        h_score = 1.0 if green_h_range[0] <= h <= green_h_range[1] else 0.0
        s_score = min(1.0, s / 255.0) if s >= min_saturation else 0.0
        v_score = min(1.0, v / 255.0) if v >= min_value else 0.0
        
        hsv_confidence = (h_score * 0.5 + s_score * 0.3 + v_score * 0.2)
        
        # RGB 보정 (녹색 우위성)
        # 오버플로 방지용 안전 합산
        total = int(r) + int(g) + int(b)
        if total > 0:
            green_ratio = g / total
            rgb_bonus = min(0.3, green_ratio) if green_ratio > 0.3 else 0.0
        else:
            rgb_bonus = 0.0
        
        final_confidence = min(1.0, hsv_confidence + rgb_bonus)
        
        return {
            'bayesian_score': hsv_confidence,
            'physics_score': rgb_bonus,
            'integrated_score': final_confidence,
            'final_confidence': final_confidence,
            'confidence_lower': max(0.0, final_confidence - 0.1),
            'confidence_upper': min(1.0, final_confidence + 0.1),
            'is_laser_candidate': final_confidence > 0.5
        }
    
    def _combine_with_depth(self, candidates: List[Dict], depth_frame: np.ndarray) -> List[Dict]:
        """깊이 정보와 RGB 후보 결합"""
        enhanced_candidates = []
        
        try:
            # 깊이 프레임 크기 조정 (필요시)
            if depth_frame.shape[:2] != self.current_gray.shape[:2]:
                depth_resized = cv2.resize(depth_frame, (self.current_gray.shape[1], self.current_gray.shape[0]))
            else:
                depth_resized = depth_frame
            
            for candidate in candidates:
                cx, cy = candidate['position']
                
                # 깊이 값 추출
                depth_mm = self._extract_valid_depth(depth_resized, cx, cy)
                
                # 스크린 모드이거나 유효한 깊이가 있으면 포함
                if self.screen_mode or (depth_mm is not None and self.depth_range_mm[0] <= depth_mm <= self.depth_range_mm[1]):
                    # 깊이가 없으면 추정 (스크린 모드)
                    if depth_mm is None or depth_mm <= 0:
                        depth_mm = 2000.0  # 2m 추정
        
        # 3D 좌표 계산
                    world_3d = self._depth_to_3d_coordinate(cx, cy, depth_mm)
                    
                    # 후보에 깊이 정보 추가
                    candidate['depth_mm'] = depth_mm
                    candidate['world_3d'] = world_3d
                    candidate['detection_method'] = 'rgb_hybrid_depth'
                    
                    enhanced_candidates.append(candidate)
            
            return enhanced_candidates
            
        except Exception as e:
            self.logger.error(f"깊이 결합 오류: {e}")
            return candidates  # 오류시 원본 반환
    
    def _is_genuine_laser_motion(self, candidate: Dict, motion_mask: np.ndarray) -> bool:
        """진짜 레이저 움직임인지 판별 (vs 일반 움직임)
        
        레이저 특성:
        - 작은 면적의 밝은 점
        - 높은 과학적 스코어
        - 특정 색상 특성 (녹색 우위)
        - 원형에 가까운 모양
        
        Args:
            candidate: 검출 후보
            motion_mask: 움직임 마스크
            
        Returns:
            True if 레이저로 판단됨, False if 일반 움직임
        """
        try:
            # 1. 면적 체크 (레이저는 작은 면적)
            area = candidate.get('area', 0)
            if area > self.laser_motion_area_threshold:
                return False  # 너무 큰 움직임은 일반 움직임
                
            # 2. 밝기 체크 (레이저는 충분히 밝아야 함)
            brightness = candidate.get('brightness', 0)
            if brightness < self.laser_brightness_threshold:
                return False
                
            # 3. 🔬 과학적 레이저 스코어 체크
            scientific_scores = candidate.get('scientific_scores', {})
            bayesian_score = scientific_scores.get('bayesian_score', 0.0)
            is_laser_candidate = candidate.get('is_green_laser', False)
            
            if not is_laser_candidate or bayesian_score < self.scientific_score_threshold:
                return False  # 과학적으로 레이저가 아님
                
            # 4. 색상 특성 체크 (녹색 우위성)
            r, g, b = candidate.get('rgb_values', (0, 0, 0))
            if r + g + b > 0:
                green_ratio = g / (r + g + b)
                if green_ratio < 0.3:  # 녹색이 30% 이하면 레이저 아님
                    return False
                    
            # 5. 원형도 체크 (레이저는 원형에 가까움)
            circularity = candidate.get('circularity', 0.0)
            if circularity < 0.3:  # 너무 길쭉하면 레이저 아님
                return False
                
            # 6. 움직임 패턴 체크 (점 형태 움직임)
            motion_detected = candidate.get('motion_detected', False)
            if motion_detected:
                cx, cy = candidate['position']
                # 움직임 영역의 크기 체크
                if (0 <= cy < motion_mask.shape[0] and 0 <= cx < motion_mask.shape[1]):
                    # 주변 3x3 영역의 움직임 체크
                    motion_area = motion_mask[max(0, cy-1):min(motion_mask.shape[0], cy+2), 
                                            max(0, cx-1):min(motion_mask.shape[1], cx+2)]
                    motion_pixels = np.sum(motion_area > 0)
                    if motion_pixels > 6:  # 3x3 영역에서 6픽셀 이상 움직임은 너무 큼
                        return False
                    
                    # 7. 🔬 시간적 일관성 체크 (노이즈 필터링)
                    if not self._check_temporal_consistency(candidate['position']):
                        return False  # 너무 불규칙한 움직임은 노이즈
                        
            self.logger.debug(f"[LASER FILTER] 진짜 레이저 검출: area={area}, brightness={brightness}, "
                            f"bayesian={bayesian_score:.3f}, green_ratio={green_ratio:.3f}")
            return True
            
        except Exception as e:
            self.logger.error(f"레이저 움직임 판별 오류: {e}")
            return False  # 오류 시 보수적으로 False

    def _apply_consecutive_detection_filter(self, candidates: List[Dict]) -> List[Dict]:
        """연속성 검증 필터 - 같은 위치에서 연속적으로 검출된 것만 인정
        
        Args:
            candidates: 현재 프레임의 후보 리스트
            
        Returns:
            연속성이 확인된 후보들만 포함한 리스트
        """
        try:
            current_time = time.time()
            validated_candidates = []
            
            # 현재 검출 이력에 추가
            for candidate in candidates:
                pos = candidate['position']
                
                # 이력에 추가
                self.detection_history.append({
                    'position': pos,
                    'timestamp': current_time,
                    'confidence': candidate.get('confidence', 0.0)
                })
            
            # 2초 이내 이력만 유지
            self.detection_history = [
                h for h in self.detection_history 
                if current_time - h['timestamp'] < 2.0
            ]
            
            # 각 후보에 대해 연속성 검증
            for candidate in candidates:
                pos = candidate['position']
                cx, cy = pos
                
                # 반경 10픽셀 내에서 최근 검출 수 확인
                consecutive_count = 0
                for history in self.detection_history:
                    hx, hy = history['position']
                    distance = np.sqrt((cx - hx)**2 + (cy - hy)**2)
                    if distance <= 10.0:  # 10픽셀 반경
                        consecutive_count += 1
                
                # 연속 검출 조건 만족 시 포함
                if consecutive_count >= self.consecutive_detection_required:
                    candidate['consecutive_count'] = consecutive_count
                    validated_candidates.append(candidate)
                    self.logger.debug(f"[CONSECUTIVE] 연속성 검증 통과: {pos}, 연속={consecutive_count}")
                else:
                    self.logger.debug(f"[CONSECUTIVE] 연속성 검증 실패: {pos}, 연속={consecutive_count}/{self.consecutive_detection_required}")
            
            return validated_candidates
            
        except Exception as e:
            self.logger.error(f"연속성 검증 필터 오류: {e}")
            return candidates  # 오류 시 원본 반환

    def _check_temporal_consistency(self, position: Tuple[int, int]) -> bool:
        """시간적 일관성 체크 (노이즈 필터링)
        
        레이저는 비교적 일관된 패턴으로 움직이므로,
        너무 불규칙한 움직임은 노이즈로 판단
        
        Args:
            position: 현재 검출 위치
            
        Returns:
            True if 일관성 있음, False if 노이즈 의심
        """
        try:
            current_time = time.time()
            cx, cy = position
            
            # 히스토리에 추가
            self.motion_history.append({
                'position': (cx, cy),
                'timestamp': current_time
            })
            
            # 최대 10개까지만 유지 (2초 이내)
            self.motion_history = [
                h for h in self.motion_history 
                if current_time - h['timestamp'] < 2.0
            ][-10:]
            
            # 히스토리가 충분하지 않으면 허용
            if len(self.motion_history) < 3:
                return True
                
            # 최근 위치들의 분산 계산
            recent_positions = [h['position'] for h in self.motion_history[-5:]]
            positions_x = [p[0] for p in recent_positions]
            positions_y = [p[1] for p in recent_positions]
            
            # 위치 분산이 너무 크면 노이즈로 판단
            variance_x = np.var(positions_x)
            variance_y = np.var(positions_y)
            total_variance = variance_x + variance_y
            
            # 분산 임계값 (픽셀²)
            max_variance = 2500  # 50픽셀 정도의 표준편차
            
            if total_variance > max_variance:
                self.logger.debug(f"[TEMPORAL] 시간적 일관성 실패: variance={total_variance:.1f}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"시간적 일관성 체크 오류: {e}")
            return True  # 오류 시 보수적으로 허용

    def _calculate_hybrid_confidence(self, candidate: Dict, has_motion_detection: bool) -> float:
        """하이브리드 신뢰도 계산"""
        try:
            # 스크린 단순 파이프라인 후보는 자체 confidence를 그대로 사용
            if candidate.get('detection_method') == 'screen_simple':
                return float(min(1.0, max(0.0, candidate.get('confidence', 0.0))))

            confidence = 0.0
            
            # 1. 밝기 점수 (40%)
            brightness = candidate.get('brightness', 0)
            brightness_score = min(1.0, max(0.0, (brightness - self.adaptive_brightness_threshold) / 100.0))
            confidence += brightness_score * self.brightness_weight
            
            # 2. 움직임 점수 (30%)
            if has_motion_detection and candidate.get('motion_detected', False):
                motion_intensity = candidate.get('motion_intensity', 0.0)
                motion_score = min(1.0, motion_intensity * 2.0)  # 강화
                confidence += motion_score * self.motion_weight
            
            # 3. 면적 점수 (20%)
            area = candidate.get('area', 0)
            optimal_area = 25.0  # 최적 면적
            area_distance = abs(area - optimal_area) / optimal_area
            area_score = max(0.2, 1.0 - area_distance)
            confidence += area_score * self.area_weight
            
            # 4. 스크린 모드 보너스 (10%) - 깊이 대신 캘리브레이션 신뢰도
            if self.calibration_mode and self.screen_mode:
                calibration_score = 1.0  # 캘리브레이션된 스크린은 항상 신뢰도 높음
            else:
                calibration_score = 0.5
            confidence += calibration_score * 0.1
            
            # 5. 원형도 보너스
            circularity = candidate.get('circularity', 0.0)
            if circularity > 0.4:
                confidence += 0.05
            
            # 🔬 6. 과학적 레이저 스코어링 (베이지안 + 물리학적)
            scientific_scores = candidate.get('scientific_scores', {})
            bayesian_confidence = scientific_scores.get('bayesian_score', 0.0)
            physics_score = scientific_scores.get('physics_score', 0.0)
            is_laser_candidate = candidate.get('is_green_laser', False)
            
            # 과학적 스코어 기반 신뢰도 계산
            if is_laser_candidate:
                # 과학적으로 레이저 후보로 판정되면 높은 신뢰도
                scientific_bonus = (bayesian_confidence * 0.4 + physics_score * 0.3)
                confidence += scientific_bonus
                self.logger.debug(f"[SCIENTIFIC] 과학적 레이저 검출: 베이지안={bayesian_confidence:.3f}, 물리학적={physics_score:.3f}")
            else:
                # 일반적인 밝은 점으로 판정되면 기존 로직 사용
                green_dominance = candidate.get('green_dominance', 0.0)
                if green_dominance > 0.5:  # 완화된 기준
                    confidence += min(0.1, green_dominance * 0.1)
            
            # 오검출 방지: 모션이 없으면 신뢰도 감소 (완전 배제하지 않음)
            if self.require_motion_for_detection and not candidate.get('motion_detected', False):
                confidence *= 0.5  # 모션 없으면 신뢰도 절반으로 감소 (배제하지 않음)
            
            # 최종 신뢰도가 임계값 이하면 0으로 설정 (스크린/총구 별도 임계 적용)
            final_confidence = min(1.0, confidence)
            threshold = self.min_confidence_threshold_gun
            if final_confidence < threshold:
                return 0.0
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"신뢰도 계산 오류: {e}")
            return 0.5  # 기본값
    
    def _extract_valid_depth(self, depth_frame: np.ndarray, x: int, y: int) -> Optional[float]:
        """견고한 깊이 값 추출"""
        try:
            if 0 <= y < depth_frame.shape[0] and 0 <= x < depth_frame.shape[1]:
                raw_depth = depth_frame[y, x]
                if 0 < raw_depth < 65535:
                    return float(raw_depth)
                
                # 주변 픽셀 탐색
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if (0 <= ny < depth_frame.shape[0] and 0 <= nx < depth_frame.shape[1]):
                            neighbor_depth = depth_frame[ny, nx]
                            if 0 < neighbor_depth < 65535:
                                return float(neighbor_depth)
            
            return None
            
        except Exception as e:
            self.logger.error(f"깊이 추출 오류: {e}")
            return None
    
    def _depth_to_3d_coordinate(self, x: int, y: int, depth_mm: float) -> Tuple[float, float, float]:
        """2D + 깊이 → 3D 월드 좌표 변환"""
        # Azure Kinect HD 추정 내부 파라미터
        fx, fy = 920.0, 920.0
        cx, cy = 960.0, 540.0
        
        world_x = (x - cx) * depth_mm / fx
        world_y = (y - cy) * depth_mm / fy
        world_z = depth_mm
        
        return (world_x, world_y, world_z)
    
    def learn_rgb_sample(self, position: Tuple[int, int], rgb_values: Tuple[int, int, int], 
                        brightness: int, depth_mm: Optional[float] = None) -> bool:
        """RGB 샘플 학습"""
        try:
            sample = {
                'position': position,
                'rgb_values': rgb_values,
                'brightness': brightness,
                'depth_mm': depth_mm,
                'timestamp': time.time()
            }
            
            self.learned_samples.append(sample)
            
            # 최대 샘플 수 유지
            if len(self.learned_samples) > self.max_samples:
                self.learned_samples.pop(0)
            
            # 적응형 임계값 업데이트
            if len(self.learned_samples) > 3:
                recent_brightness = [s['brightness'] for s in self.learned_samples[-10:]]
                avg_brightness = sum(recent_brightness) / len(recent_brightness)
                self.adaptive_brightness_threshold = max(80, int(avg_brightness * 0.8))
            
            self.logger.info(f"RGB 샘플 학습: 밝기={brightness}, 임계값={self.adaptive_brightness_threshold}")
            return True
            
        except Exception as e:
            self.logger.error(f"RGB 학습 오류: {e}")
            return False
    
    def get_detection_stats(self) -> Dict:
        """검출 통계 반환"""
        return {
            'learned_samples': len(self.learned_samples),
            'adaptive_threshold': self.adaptive_brightness_threshold,
            'motion_threshold': self.motion_threshold,
            'area_range': (self.min_laser_area, self.max_laser_area),
            'depth_range_mm': self.depth_range_mm,
            'screen_mode': self.screen_mode,
            'stats': self.stats.copy()
        }
    
    def reset_learning_data(self):
        """학습 데이터 초기화"""
        self.learned_samples.clear()
        self.adaptive_brightness_threshold = self.brightness_threshold
        self.previous_frame = None
        self.detection_history.clear()
        self.stats = {
            'total_detections': 0,
            'motion_detections': 0,
            'brightness_detections': 0,
            'hybrid_detections': 0
        }
        print("[INFO] RGB 학습 데이터 초기화 완료")
    
    def adjust_parameters(self, param_name: str, value: float):
        """파라미터 동적 조정"""
        if param_name == 'brightness_threshold':
            self.adaptive_brightness_threshold = max(50, min(255, int(value)))
        elif param_name == 'motion_threshold':
            self.motion_threshold = max(10, min(100, int(value)))
        elif param_name == 'min_area':
            self.min_laser_area = max(1, int(value))
        elif param_name == 'max_area':
            self.max_laser_area = max(self.min_laser_area + 1, int(value))
        elif param_name == 'screen_ratio':
            # 0.40 ~ 0.90 범위로 제한
            self.screen_ratio_threshold = float(max(0.40, min(0.90, value)))
        elif param_name == 'screen_exg':
            # 0 ~ 255 범위
            self.screen_exg_threshold = int(max(0, min(255, value)))
        elif param_name == 'screen_contrast':
            self.screen_contrast_threshold = int(max(0, min(255, value)))
        elif param_name == 'screen_log':
            self.screen_log_threshold = int(max(0, min(255, value)))
        elif param_name == 'screen_temporal_needed':
            self.screen_temporal_needed = int(max(1, min(self.screen_temporal_window, value)))
        elif param_name == 'screen_temporal_radius':
            self.screen_temporal_radius = int(max(0, min(20, value)))
        
        print(f"[INFO] 파라미터 조정: {param_name} = {value}")