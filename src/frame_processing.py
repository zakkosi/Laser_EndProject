"""
Virtra 프레임 처리 모듈 (Frame Processing Module)
원본 adaptive_laser_detector.py에서 프레임 차이 검출 시스템 추출

핵심 기능:
- 프레임 버퍼 관리 (update_frame_buffer)
- 움직임 영역 검출 (detect_motion_regions)
- 프레임 차이 기반 검출 (combine_motion_and_color_detection)
- 레이저 크기 기반 모션 필터링 (filter_motion_by_laser_size)
- 프레임 차이 통계 관리
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MotionDetectionResult:
    """모션 검출 결과"""
    motion_detected: bool
    motion_mask: np.ndarray
    motion_area: int
    motion_regions: List[Tuple[int, int, int, int]]  # (x, y, w, h)
    processing_time_ms: float


class FrameProcessor:
    """
    프레임 차이 기반 검출 시스템
    
    담당 기능:
    - 프레임 버퍼 관리 및 업데이트
    - 움직임 영역 검출 및 분석
    - 프레임 차이와 HSV 검출 결합
    - 레이저 크기 기반 모션 필터링
    - 프레임 차이 통계 관리
    """
    
    def __init__(self):
        """프레임 처리 모듈 초기화"""
        # 프레임 차이 기반 동적 검출 시스템 (원본과 동일)
        self.enable_frame_diff = True  # 프레임 차이 검출 활성화
        self.previous_frames = []  # 이전 프레임들 저장 (최대 3프레임)
        self.frame_diff_threshold = 35  # 프레임 차이 임계값
        self.motion_buffer_size = 3  # 이전 프레임 버퍼 크기
        self.min_motion_area = 5  # 최소 움직임 영역 크기
        self.max_motion_area = 300  # 최대 움직임 영역 크기
        self.temporal_consistency = 2  # 연속 프레임에서 검출되어야 하는 횟수
        
        # 레이저 특화 파라미터 (원본과 동일)
        self.laser_motion_radius = 15  # 레이저 주변 움직임 검사 반경
        self.motion_cluster_max_size = 200  # 움직임 클러스터 최대 크기
        
        # 프레임 차이 통계 (원본과 동일)
        self.frame_diff_stats = {
            'motion_detections': 0,
            'static_rejections': 0,
            'combined_success': 0
        }
        
        # 로깅
        self.logger = logging.getLogger(__name__)
        
        # 현재 모션 마스크 (디버그용)
        self.current_motion_mask = None
        
        print("[INFO] 프레임 처리 모듈 초기화 완료")
    
    def update_frame_buffer(self, frame: np.ndarray):
        """이전 프레임 버퍼 업데이트 (원본 _update_frame_buffer)"""
        if frame is None:
            return
        
        # 그레이스케일 변환 (빠른 처리)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 프레임 버퍼에 추가
        self.previous_frames.append(gray_frame.copy())
        
        # 버퍼 크기 제한 (메모리 절약)
        if len(self.previous_frames) > self.motion_buffer_size:
            self.previous_frames.pop(0)
    
    def detect_motion(self, current_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        간단한 모션 검출 (호환성 메서드)
        기존 detect_motion_regions를 호출하여 motion_mask만 반환
        """
        try:
            result = self.detect_motion_regions(current_frame)
            return result.motion_mask if result.motion_detected else None
        except Exception as e:
            self.logger.error(f"모션 검출 실패: {e}")
            return None
    
    def detect_motion_regions(self, current_frame: np.ndarray) -> MotionDetectionResult:
        """
        프레임 차이로 레이저 크기 움직임/변화 영역 검출 (원본 _detect_motion_regions)
        """
        start_time = time.time()
        
        # 기본 결과 초기화
        result = MotionDetectionResult(
            motion_detected=False,
            motion_mask=np.zeros(current_frame.shape[:2], dtype=np.uint8),
            motion_area=0,
            motion_regions=[],
            processing_time_ms=0.0
        )
        
        if len(self.previous_frames) < 2:
            # 초기 프레임들 - 전체 영역 반환
            result.motion_mask = np.ones(current_frame.shape[:2], dtype=np.uint8) * 255
            result.motion_detected = True
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        
        try:
            # 현재 프레임을 그레이스케일로 변환
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            
            # 이전 프레임과의 차이 계산 (가장 최근 프레임과 비교)
            prev_gray = self.previous_frames[-1]
            
            # 절댓값 차이 계산 (OpenCV 최적화)
            frame_diff = cv2.absdiff(current_gray, prev_gray)
            
            # 임계값 적용하여 변화 영역 추출
            _, motion_mask = cv2.threshold(frame_diff, self.frame_diff_threshold, 255, cv2.THRESH_BINARY)
            
            # 추가 최적화: 두 번째 이전 프레임과도 비교 (더 확실한 움직임)
            if len(self.previous_frames) >= 2:
                prev_gray2 = self.previous_frames[-2]
                frame_diff2 = cv2.absdiff(current_gray, prev_gray2)
                _, motion_mask2 = cv2.threshold(frame_diff2, self.frame_diff_threshold, 255, cv2.THRESH_BINARY)
                
                # 두 차이 마스크 중 하나라도 변화가 있으면 움직임으로 간주
                motion_mask = cv2.bitwise_or(motion_mask, motion_mask2)
            
            # 레이저 크기에 맞는 움직임 영역만 추출
            motion_mask = self.filter_motion_by_laser_size(motion_mask)
            
            # 모션 영역 분석
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_regions = []
            total_motion_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_motion_area <= area <= self.max_motion_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    valid_regions.append((x, y, w, h))
                    total_motion_area += area
            
            # 통계 업데이트
            motion_pixels = np.sum(motion_mask > 0)
            if motion_pixels > self.min_motion_area:
                self.frame_diff_stats['motion_detections'] += 1
                result.motion_detected = True
                result.motion_mask = motion_mask
                result.motion_area = total_motion_area
                result.motion_regions = valid_regions
            else:
                self.frame_diff_stats['static_rejections'] += 1
                result.motion_mask = np.zeros(current_frame.shape[:2], dtype=np.uint8)
            
            # 현재 모션 마스크 저장 (디버그용)
            self.current_motion_mask = result.motion_mask
            
        except Exception as e:
            self.logger.warning(f"움직임 검출 오류: {e}")
            # 오류시 전체 영역 반환 (안전장치)
            result.motion_mask = np.ones(current_frame.shape[:2], dtype=np.uint8) * 255
            result.motion_detected = True
        
        finally:
            result.processing_time_ms = (time.time() - start_time) * 1000
        
        return result
    
    def filter_motion_by_laser_size(self, motion_mask: np.ndarray) -> np.ndarray:
        """
        레이저 크기에 맞는 움직임 영역만 추출 (원본 _filter_motion_by_laser_size)
        """
        try:
            # 윤곽선으로 움직임 영역들을 분리
            contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 레이저 크기에 맞는 움직임만 남기기
            filtered_mask = np.zeros_like(motion_mask)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # 레이저 크기 조건: 5-300픽셀
                if self.min_motion_area <= area <= self.max_motion_area:
                    # 추가 조건: 원형성 검사 (레이저는 대략 원형)
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        
                        # 원형성이 0.2 이상인 것만 (너무 길쭉한 모양 제거)
                        if circularity > 0.2:
                            # 컨투어를 마스크에 그리기
                            cv2.fillPoly(filtered_mask, [contour], 255)
                        # 원형성이 낮아도 면적이 작으면 허용 (작은 레이저 포인트)
                        elif area <= 50:
                            cv2.fillPoly(filtered_mask, [contour], 255)
            
            # 추가 필터링: 노이즈 제거 (최소한만)
            if np.sum(filtered_mask > 0) > 0:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                filtered_mask = cv2.morphologyEx(filtered_mask, cv2.MORPH_CLOSE, kernel)
            
            return filtered_mask
            
        except Exception as e:
            self.logger.warning(f"움직임 필터링 오류: {e}")
            return motion_mask  # 오류시 원본 반환
    
    def combine_motion_and_color_detection(self, frame: np.ndarray, motion_mask: np.ndarray, 
                                         laser_detector_core) -> List[Dict]:
        """
        움직임 영역 + HSV 검출 결합 (원본 _combine_motion_and_color_detection)
        
        Args:
            frame: 입력 프레임
            motion_mask: 모션 마스크
            laser_detector_core: 레이저 검출 핵심 모듈
            
        Returns:
            검출된 레이저 후보 목록
        """
        candidates = []
        
        try:
            # 1단계: 움직임이 있는 영역만 HSV 검출 수행
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 동적 HSV 프로필 적용
            dynamic_profiles = laser_detector_core.get_dynamic_hsv_profiles()
            
            for profile_name in laser_detector_core.active_profiles:
                if profile_name not in dynamic_profiles:
                    continue
                
                profile = dynamic_profiles[profile_name]
                
                # HSV 마스크 생성
                lower_bound = np.array([profile['h_min'], profile['s_min'], profile['v_min']])
                upper_bound = np.array([profile['h_max'], profile['s_max'], 255])
                color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
                
                # 핵심: 움직임 마스크와 색상 마스크 결합
                combined_mask = cv2.bitwise_and(color_mask, motion_mask)
                
                # 결합된 마스크에서 윤곽선 찾기
                contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < 0.5:  # 최소 면적
                        continue
                    
                    # 중심점 계산
                    M = cv2.moments(contour)
                    if M['m00'] == 0:
                        continue
                    
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # 경계 확인
                    if not (0 <= cx < frame.shape[1] and 0 <= cy < frame.shape[0]):
                        continue
                    
                    # HSV 값 추출
                    h, s, v = hsv[cy, cx]
                    brightness = float(gray[cy, cx])
                    
                    # 레이저 점수 계산 (움직임 보너스 추가)
                    base_score = laser_detector_core.calculate_laser_score(
                        brightness, int(h), int(s), int(v), area
                    )
                    
                    # 움직임 보너스: 움직임이 감지된 영역은 신뢰도 +0.05 (완화됨)
                    motion_bonus = 0.05 if motion_mask[cy, cx] > 0 else 0
                    final_score = min(1.0, base_score + motion_bonus)
                    
                    if final_score > 0.3:  # 상향된 임계값 (오검출 방지)
                        candidates.append({
                            'position': (cx, cy),
                            'hsv': (int(h), int(s), int(v)),
                            'brightness': brightness,
                            'area': area,
                            'confidence': final_score,
                            'method': f"{profile_name}_motion",  # 움직임 결합 표시
                            'motion_detected': True
                        })
            
            # 성공 통계 업데이트
            if candidates:
                self.frame_diff_stats['combined_success'] += 1
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"결합 검출 오류: {e}")
            return []
    
    def get_motion_stats(self) -> Dict:
        """프레임 차이 통계 반환"""
        total_motion = self.frame_diff_stats['motion_detections'] + self.frame_diff_stats['static_rejections']
        motion_rate = (self.frame_diff_stats['motion_detections'] / max(1, total_motion)) * 100
        
        return {
            'enabled': self.enable_frame_diff,
            'motion_rate': motion_rate,
            'motion_detections': self.frame_diff_stats['motion_detections'],
            'static_rejections': self.frame_diff_stats['static_rejections'],
            'combined_success': self.frame_diff_stats['combined_success'],
            'frame_diff_threshold': self.frame_diff_threshold,
            'max_motion_area': self.max_motion_area,
            'min_motion_area': self.min_motion_area,
            'motion_buffer_size': self.motion_buffer_size
        }
    
    def toggle_frame_diff(self) -> bool:
        """프레임 차이 검출 ON/OFF 토글"""
        self.enable_frame_diff = not self.enable_frame_diff
        status = "ON" if self.enable_frame_diff else "OFF"
        print(f"🎯 프레임 차이 검출: {status}")
        return self.enable_frame_diff
    
    def adjust_frame_diff_threshold(self, increase: bool = True):
        """프레임 차이 임계값 조정"""
        if increase:
            self.frame_diff_threshold = min(80, self.frame_diff_threshold + 5)
        else:
            self.frame_diff_threshold = max(15, self.frame_diff_threshold - 5)
        
        print(f"📊 프레임 차이 임계값: {self.frame_diff_threshold} (15-80)")
    
    def adjust_max_motion_area(self, increase: bool = True):
        """최대 움직임 영역 크기 조정"""
        if increase:
            self.max_motion_area = min(500, self.max_motion_area + 50)
        else:
            self.max_motion_area = max(100, self.max_motion_area - 50)
        
        print(f"📊 최대 움직임 크기: {self.max_motion_area}픽셀")
    
    def get_current_motion_mask(self) -> Optional[np.ndarray]:
        """현재 모션 마스크 반환 (디버그용)"""
        return self.current_motion_mask
    
    def reset_statistics(self):
        """통계 초기화"""
        self.frame_diff_stats = {
            'motion_detections': 0,
            'static_rejections': 0,
            'combined_success': 0
        }
        print("📊 프레임 차이 통계 초기화 완료")
    
    def is_frame_diff_enabled(self) -> bool:
        """프레임 차이 검출 활성화 상태 반환"""
        return self.enable_frame_diff
    
    def get_frame_buffer_status(self) -> Dict:
        """프레임 버퍼 상태 반환"""
        return {
            'buffer_size': len(self.previous_frames),
            'max_buffer_size': self.motion_buffer_size,
            'ready': len(self.previous_frames) >= 2
        }
    
    def print_detailed_stats(self):
        """상세 통계 출력"""
        stats = self.get_motion_stats()
        
        print(f"\n🎯 프레임 차이 검출 상세 통계:")
        print(f"  상태: {'ON' if stats['enabled'] else 'OFF'}")
        print(f"  움직임 감지율: {stats['motion_rate']:.1f}%")
        print(f"  정적 영역 제거: {stats['static_rejections']}회")
        print(f"  움직임+색상 결합 성공: {stats['combined_success']}회")
        print(f"  프레임 차이 임계값: {stats['frame_diff_threshold']} (15-80)")
        print(f"  최대 움직임 크기: {stats['max_motion_area']}픽셀")
        print(f"  최소 움직임 크기: {stats['min_motion_area']}픽셀")
        print(f"  프레임 버퍼 크기: {stats['motion_buffer_size']}프레임")
        
        # 프레임 버퍼 상태
        buffer_status = self.get_frame_buffer_status()
        print(f"  프레임 버퍼 상태: {buffer_status['buffer_size']}/{buffer_status['max_buffer_size']} {'(준비됨)' if buffer_status['ready'] else '(초기화중)'}")
    
    def create_debug_mask_display(self, frame: np.ndarray, color_mask: np.ndarray) -> np.ndarray:
        """디버그용 마스크 시각화 생성"""
        if self.current_motion_mask is None:
            return np.zeros_like(frame)
        
        # 움직임 + 색상 결합 마스크
        combined_mask = cv2.bitwise_and(color_mask, self.current_motion_mask)
        
        # 3채널 마스크 생성 (시각화 개선)
        motion_only = cv2.bitwise_and(self.current_motion_mask, cv2.bitwise_not(color_mask))
        color_only = cv2.bitwise_and(color_mask, cv2.bitwise_not(self.current_motion_mask))
        
        # 컬러 마스크: 빨강(움직임만), 초록(색상만), 노랑(둘다)
        display_mask = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        display_mask[:,:,2] = motion_only  # 빨강: 움직임만
        display_mask[:,:,1] = color_only   # 초록: 색상만
        display_mask[:,:,1] += combined_mask  # 노랑: 움직임+색상
        display_mask[:,:,2] += combined_mask
        
        # 텍스트 정보 추가
        cv2.putText(display_mask, "Red: Motion Only", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display_mask, "Green: Color Only", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display_mask, "Yellow: Motion+Color", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return display_mask
    
    def get_latest_motion_mask(self) -> Optional[np.ndarray]:
        """최신 모션 마스크 반환 (디버그 시각화용)"""
        return self.current_motion_mask 