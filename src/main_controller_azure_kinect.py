"""
Virtra 메인 컨트롤러 모듈 (Azure Kinect 통합 버전)
기존 모듈화 구조를 유지하면서 Azure Kinect DK 지원 추가

핵심 변경사항:
- camera_manager.py를 사용한 통합 카메라 관리
- Azure Kinect DK (단일/듀얼) 지원 추가
- 기존 laser_detector_core, frame_processing, screen_mapping 모듈 그대로 유지
- 깊이 정보 활용 강화
"""

import cv2
import numpy as np
import time
import logging
import json
import socket
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 기존 모듈화된 컴포넌트들 (그대로 유지)
from laser_detector_core import LaserDetectorCore, LaserDetectionResult
from frame_processing import FrameProcessor
from screen_mapping import ScreenMapper
from gun_muzzle_detector import MuzzleDetector
from body_tracking_worker import BodyTrackingWorker

# Enhanced Laser Detector (선택적 사용)
try:
    from enhanced_laser_detector import EnhancedLaserDetector
    ENHANCED_DETECTOR_AVAILABLE = True
    print("[INFO] Enhanced Laser Detector (Modified CHT) Available")
except ImportError:
    ENHANCED_DETECTOR_AVAILABLE = False
    print("[INFO] Enhanced Laser Detector NOT Available - Using Standard Mode")

# 새로운 카메라 관리 모듈
from camera_manager import (
    CameraManager, CameraType, CameraConfig, CameraFrame,
    create_webcam_manager, create_zed_manager, 
    create_azure_kinect_manager, create_dual_azure_kinect_manager
)


class MainControllerAzureKinect:
    """
    Azure Kinect 통합 메인 시스템 컨트롤러
    
    담당 기능:
    - 다양한 카메라 타입 지원 (웹캠, ZED, Azure Kinect)
    - 기존 모듈화 구조 유지
    - 깊이 정보 활용 강화
    - 듀얼 카메라 시스템 지원
    """
    
    def __init__(self, camera_type: str = "azure_kinect", **camera_kwargs):
        """메인 컨트롤러 초기화"""
        self.camera_type = camera_type.lower()
        self.camera_kwargs = camera_kwargs
        
        # 카메라 관리자 초기화
        self.camera_manager = self._create_camera_manager()
        self.secondary_camera_manager = None  # 듀얼 모드용
        
        # 기존 모듈화된 컴포넌트들 (Azure Kinect 최적화)
        self.laser_core = LaserDetectorCore()
        # 듀얼 모드 분리: 카메라별 코어를 분리하여 상태 간섭(이전 프레임, 모션 히스토리, ROI)을 제거
        self.laser_core_screen = LaserDetectorCore()
        self.laser_core_gun = LaserDetectorCore()
        # 듀얼 모드 안정화를 위한 카메라별 프레임 프로세서 분리
        self.frame_processor = FrameProcessor()
        self.frame_processor.enable_frame_diff = True  # 핵심: 프레임 차이 검출 활성화 (속성 설정)
        self.frame_processor_primary = FrameProcessor()
        self.frame_processor_primary.enable_frame_diff = True
        self.frame_processor_secondary = FrameProcessor()
        self.frame_processor_secondary.enable_frame_diff = True
        self.screen_mapper = ScreenMapper()
        # 보조 카메라 총구 전용 검출기
        self.muzzle_detector = MuzzleDetector()
        
        # Body Tracking 워커 (총구 ROI 동적 업데이트용)
        self.bt_worker = None
        self.enable_bt = bool(camera_kwargs.get('enable_bt', False))
        
        # Enhanced Detector (선택적 사용) - 기본은 비활성화
        self.use_enhanced_detector = camera_kwargs.get('use_enhanced_detector', False)
        self.enhanced_laser_core = None
        
        if ENHANCED_DETECTOR_AVAILABLE and self.use_enhanced_detector:
            # Enhanced Detector 사용 (기본은 CHT 비활성화 상태)
            enable_cht = camera_kwargs.get('enable_cht', False)
            self.enhanced_laser_core = EnhancedLaserDetector(enable_cht=enable_cht)
            print(f"[INFO] Enhanced Detector Active (CHT: {'ON' if enable_cht else 'OFF'})")
        else:
            print("[INFO] Standard Laser Detector Active")
        
        # 시스템 상태
        self.is_running = False
        self.current_frame = None
        self.current_depth = None
        self.last_clicked_rgbd = None  # W키 샘플링용 저장소
        
        # 듀얼 카메라용 보조 프레임
        self.secondary_frame = None
        self.secondary_depth = None
        self._last_secondary_result = None  # 보조 카메라 검출 결과 저장용
        
        self.detection_results = []
        
        # 성능 통계
        self.stats = {
            'frames_processed': 0,
            'detections': 0,
            'depth_detections': 0,
            'triangulated_points': 0,
            'start_time': time.time(),
            'fps': 0.0
        }
        
        # UI 설정
        self.show_debug = False
        self.show_depth = True  # 깊이 맵 기본 표시
        self.show_calibration = False  # 캘리브레이션 모드
        
        # RGB 디버깅용 변수 추가
        self.current_image = None  # 현재 BGR 이미지
        self.current_gray = None   # 현재 그레이스케일 이미지
        
        # 로깅
        self.logger = logging.getLogger(__name__)
        
        print(f"[INFO] Azure Kinect Main Controller Initialized")
        print(f"[INFO] Camera Type: {self.camera_type}")
        
        # Azure Kinect 전용 설정
        self.azure_kinect_config = {
            'use_depth_filtering': True,
            'depth_range_mm': (500, 3000),
            'triangulation_enabled': camera_type == "dual_azure_kinect",
            'subpixel_accuracy': True,
            
            # 3D 벡터 생성을 위한 RGB 하이브리드 설정 (레이저 밝기 기반)
            'primary_brightness_threshold': 120,   # Camera #1: 스크린 레이저 포인트 밝기
            'secondary_brightness_threshold': 120  # Camera #2: 레이저 포인터 총구 밝기
        }

        # 스크린 평면(월드) 정의: 캘리브레이션 파일을 우선 사용
        self.screen_plane = self._load_screen_plane_from_calibration()
        
        # 캘리브레이션 설정 (에피폴라인 지오메트리용)
        self.calibration_config = {
            'chessboard_size': (8, 6),  # 체스보드 내부 코너 개수 (가로, 세로) - 사용자 이미지 기준
            'square_size': 80.0,  # 체스보드 정사각형 크기 (mm) - A1 사이즈 기준
            'min_corners_detected': 20,  # 최소 검출 코너 수
            'subpix_criteria': (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        }
        
        # 성능 최적화 설정
        self.calibration_only_mode = False  # 캘리브레이션 전용 모드 (레이저 검출 비활성화)
        self.chessboard_detection_interval = 2  # 체스보드 검출 간격 (매 N프레임마다) - 실시간 표시
        self.chessboard_frame_counter = 0  # 체스보드 검출 프레임 카운터
        
        # 듀얼 카메라 성능 최적화
        self.frame_skip_counter = 0  # 프레임 스킵 카운터
        self.frame_skip_interval = 1  # 매 N프레임마다만 처리 (1=모든 프레임, 2=격프레임)
        self.last_frame_time = time.time()  # FPS 측정용
        
        # 듀얼 모드 최적화: ROI 자동 설정
        if camera_type == "dual_azure_kinect":
            self._setup_dual_mode_roi()
        
        # 캘리브레이션 데이터 저장용
        self.calibration_data = {
            'primary_corners': [],      # 1번 카메라 코너 좌표들 (각 이미지별)
            'secondary_corners': [],    # 2번 카메라 코너 좌표들 (각 이미지별)
            'object_points': [],        # 3D 월드 좌표 (각 이미지별)
            'primary_images': [],       # 1번 카메라 원본 이미지들
            'secondary_images': [],     # 2번 카메라 원본 이미지들
            'image_count': 0,
            'calibration_complete': False,
            'min_images_required': 20,  # 최소 필요 이미지 수 (품질 향상을 위해 증가)
            'max_images_allowed': 30,   # 최대 수집 이미지 수
            'primary_detected': False,
            'secondary_detected': False,
            'last_capture_time': 0      # 마지막 캡처 시간 (중복 방지)
        }
        
        # 캘리브레이션 결과 저장용
        self.calibration_results = {
            'primary_camera_matrix': None,
            'primary_distortion': None,
            'secondary_camera_matrix': None,
            'secondary_distortion': None,
            'rotation_matrix': None,
            'translation_vector': None,
            'essential_matrix': None,
            'fundamental_matrix': None,
            'rectification_maps': None,
            'calibration_error': None,
            'calibration_date': None
        }
    
    def _create_camera_manager(self) -> CameraManager:
        """카메라 관리자 생성"""
        try:
            if self.camera_type == "webcam":
                return create_webcam_manager(
                    device_id=self.camera_kwargs.get('device_id', 0)
                )
            elif self.camera_type == "zed":
                return create_zed_manager()
            elif self.camera_type == "azure_kinect":
                return create_azure_kinect_manager(
                    device_id=self.camera_kwargs.get('device_id', 0),
                    use_4k=self.camera_kwargs.get('use_4k', False)
                )
            elif self.camera_type == "dual_azure_kinect":
                return create_dual_azure_kinect_manager(
                    use_4k=self.camera_kwargs.get('use_4k', True)  # 듀얼은 기본 4K
                )
            else:
                raise ValueError(f"지원하지 않는 카메라 타입: {self.camera_type}")
                
        except Exception as e:
            # 로거가 아직 초기화되지 않을 수 있으므로 직접 출력
            print(f"[ERROR] Camera Manager Creation Failed: {e}")
            raise
    
    def initialize_system(self) -> bool:
        """시스템 초기화"""
        try:
            print("[INFO] Initializing System...")
            
            # 카메라 초기화
            if not self.camera_manager.initialize():
                print("[ERROR] Camera Initialization Failed")
                return False
            
            # 카메라 정보 출력
            camera_info = self.camera_manager.get_camera_info()
            print(f"[INFO] Camera Initialized: {camera_info['camera_type']}")
            print(f"[INFO] Resolution: {camera_info['resolution']}")
            print(f"[INFO] FPS: {camera_info['fps']}")
            print(f"[INFO] Depth Sensor: {'Supported' if camera_info['use_depth'] else 'Not Supported'}")
            
            # Azure Kinect 추가 정보
            if 'use_4k' in camera_info:
                print(f"[INFO] 4K Mode: {'Enabled' if camera_info['use_4k'] else 'Disabled'}")
            if 'secondary_available' in camera_info:
                print(f"[INFO] Dual Mode: {'2 Devices Connected' if camera_info['secondary_available'] else '1 Device Only'}")
            
            # 기존 모듈들 초기화 (Unity 통신 포트 확인)
            unity_port = self._get_unity_port()
            print(f"[INFO] Unity Port: {unity_port}")
            
            # Body Tracking 워커 시작 (듀얼 모드에서 보조 카메라 기준)
            try:
                if self.camera_type == "dual_azure_kinect" and self.enable_bt:
                    self.bt_worker = BodyTrackingWorker(camera_id=1, target_fps=12)
                    # 공유 캡처 모드 활성화: CameraManager에서 제공하는 raw_capture 사용
                    self.bt_worker.external_capture_mode = True
                    started = self.bt_worker.start()
                    print(f"[BT] Body Tracking: {'활성화' if started else '비활성화'}")
                else:
                    print("[BT] Body Tracking: 비활성화 (플래그 미설정)")
                    self.bt_worker = None
            except Exception as e:
                print(f"[WARN] Body Tracking 시작 실패: {e}")
            
            print("[INFO] System Initialization Complete!")
            return True
            
        except Exception as e:
            self.logger.error(f"시스템 초기화 실패: {e}")
            return False

    def _load_screen_plane_from_calibration(self) -> Optional[Dict]:
        """캘리브레이션에서 스크린 평면(법선 n, 평면점 P0)을 로드"""
        try:
            with open('azure_kinect_3d_calibration.json', 'r') as f:
                calib = json.load(f)
            plane = calib.get('screen_plane', {})
            # 기대 구조: {'point': [x,y,z], 'normal': [nx,ny,nz]}
            if 'point' in plane and 'normal' in plane:
                return {
                    'point': tuple(map(float, plane['point'])),
                    'normal': tuple(map(float, plane['normal']))
                }
            # 폴백: corners_3d로 평면 계산
            corners = plane.get('corners_3d') if isinstance(plane, dict) else None
            if corners and len(corners) >= 3:
                import numpy as np
                p0 = np.array(corners[0], dtype=np.float64)
                p1 = np.array(corners[1], dtype=np.float64)
                p2 = np.array(corners[2], dtype=np.float64)
                n = np.cross(p1 - p0, p2 - p0)
                norm = np.linalg.norm(n)
                if norm > 1e-9:
                    n = n / norm
                    return {'point': (float(p0[0]), float(p0[1]), float(p0[2])),
                            'normal': (float(n[0]), float(n[1]), float(n[2]))}
        except Exception:
            pass
        return None

    def _intersect_screen_plane_with_primary_ray(self, pixel_uv: Tuple[int,int]) -> Optional[Tuple[float,float,float]]:
        """1번 카메라 픽셀로부터 생성한 광선과 스크린 평면의 교점을 계산"""
        try:
            if self.screen_plane is None:
                return None
            # 내부 파라미터(HD 추정) – 실제 값이 있다면 교체
            fx, fy = 1000.0, 1000.0
            cx, cy = 960.0, 540.0
            u, v = float(pixel_uv[0]), float(pixel_uv[1])
            # 카메라 좌표계 방향 벡터(d_cam) 계산 후 정규화
            d_cam = np.array([(u - cx)/fx, (v - cy)/fy, 1.0], dtype=np.float64)
            d_cam = d_cam / np.linalg.norm(d_cam)
            # 1번 카메라 원점을 월드 원점으로 가정(외부파라미터 미사용 케이스)
            Cw = np.zeros(3, dtype=np.float64)
            Dw = d_cam
            P0 = np.array(self.screen_plane['point'], dtype=np.float64)
            n = np.array(self.screen_plane['normal'], dtype=np.float64)
            denom = np.dot(Dw, n)
            if abs(denom) < 1e-6:
                return None
            t = np.dot(P0 - Cw, n) / denom
            if t <= 0:
                return None
            Tw = Cw + t * Dw
            # mm 단위 사용(기본 캘리브레이션 단위에 맞춘다고 가정)
            return (float(Tw[0]), float(Tw[1]), float(Tw[2]))
        except Exception:
            return None
    
    def _get_unity_port(self) -> int:
        """카메라 타입에 따른 Unity 포트 반환"""
        port_mapping = {
            'webcam': 12345,
            'zed': 9998,
            'azure_kinect': 9999,
            'dual_azure_kinect': 9997
        }
        return port_mapping.get(self.camera_type, 12345)
    
    def detect_laser_with_depth(self, color_frame: np.ndarray, depth_frame: Optional[np.ndarray]) -> LaserDetectionResult:
        """
        깊이 정보를 활용한 레이저 검출 (기존 laser_detector_core + 깊이 정보)
        """
        start_time = time.time()
        
        # RGB 하이브리드 detector 사용 (Enhanced 일시 비활성화)
        # Enhanced Detector는 아직 RGB 시스템과 호환되지 않음
        candidates = self.laser_core.detect_laser_candidates(color_frame, depth_frame)
        
        # 가장 좋은 후보 선택
        if not candidates:
            return LaserDetectionResult(
                detected=False,
                confidence=0.0,
                position=(0, 0),
                rgb_values=(0, 0, 0),
                brightness=0,
                detection_method="none",
                detection_time_ms=(time.time() - start_time) * 1000,
                screen_coordinate=(0.0, 0.0),
                unity_coordinate=(0.0, 0.0)
            )
        
        # 가장 높은 신뢰도의 후보 선택
        best_candidate = max(candidates, key=lambda c: c.get('confidence', 0))
        
        # 깊이 정보가 있는 경우 추가 처리
        if (depth_frame is not None and 
            self.azure_kinect_config['use_depth_filtering'] and 
            best_candidate.get('confidence', 0) > 0):
            
            x, y = best_candidate['position']
            
            try:
                # 해상도 불일치 해결: 4K 좌표를 깊이 맵 크기에 맞게 스케일링
                depth_height, depth_width = depth_frame.shape[:2]
                
                # 4K 해상도에서 깊이 맵 해상도로 좌표 변환 (안전한 clamp 적용)
                if self.camera_kwargs.get('use_4k', False):
                    color_height, color_width = 2160, 3840
                    scaled_x = max(0, min(int(x * depth_width / color_width), depth_width - 1))
                    scaled_y = max(0, min(int(y * depth_height / color_height), depth_height - 1))
                else:
                    scaled_x = max(0, min(x, depth_width - 1))
                    scaled_y = max(0, min(y, depth_height - 1))
                
                # 디버그 로깅 (필요시)
                if x != scaled_x or y != scaled_y:
                    self.logger.debug(f"좌표 스케일링: ({x}, {y}) → ({scaled_x}, {scaled_y})")
                
                # 좌표 범위 검증 (이제 항상 통과해야 함)
                if not (0 <= scaled_x < depth_width and 0 <= scaled_y < depth_height):
                    self.logger.warning(f"깊이 좌표 범위 초과: ({scaled_x}, {scaled_y}) vs ({depth_width}, {depth_height})")
                    return LaserDetectionResult(
                        detected=best_candidate.get('confidence', 0) > 0,
                        confidence=best_candidate.get('confidence', 0),
                        position=best_candidate.get('position', (0, 0)),
                        rgb_values=best_candidate.get('rgb_values', (0, 0, 0)),
                        brightness=best_candidate.get('brightness', 0),
                        detection_method=f"{best_candidate.get('detection_method', 'unknown')}+depth_out_of_bounds",
                        detection_time_ms=(time.time() - start_time) * 1000,
                        screen_coordinate=best_candidate.get('screen_coordinate', (0.0, 0.0)),
                        unity_coordinate=best_candidate.get('unity_coordinate', (0.0, 0.0))
                    )
                
                # 깊이 값 추출 (스케일링된 좌표 사용)
                depth_mm = float(depth_frame[scaled_y, scaled_x])
                
                # 유효 깊이 범위 확인
                min_depth, max_depth = self.azure_kinect_config['depth_range_mm']
                
                if min_depth <= depth_mm <= max_depth:
                    # 깊이 기반 신뢰도 보너스
                    depth_confidence_bonus = 0.2
                    enhanced_confidence = min(1.0, best_candidate.get('confidence', 0) + depth_confidence_bonus)
                    
                    # 3D 월드 좌표 계산 (원본 4K 좌표 사용)
                    world_coords = self._pixel_to_world_coordinates(x, y, depth_mm)
                    
                    self.stats['depth_detections'] += 1
                    
                    return LaserDetectionResult(
                        detected=True,
                        confidence=enhanced_confidence,
                        position=(x, y),
                        rgb_values=best_candidate.get('rgb_values', (0, 0, 0)),
                        brightness=best_candidate.get('brightness', 0),
                        detection_method=f"{best_candidate.get('detection_method', 'unknown')}+depth",
                        detection_time_ms=(time.time() - start_time) * 1000,
                        screen_coordinate=best_candidate.get('screen_coordinate', (0.0, 0.0)),
                        unity_coordinate=best_candidate.get('unity_coordinate', (0.0, 0.0))
                    )
                else:
                    # 깊이가 유효 범위를 벗어나면 검출 무효화
                    return LaserDetectionResult(
                        detected=False,
                        confidence=0.0,
                        position=(0, 0),
                        rgb_values=(0, 0, 0),
                        brightness=0.0,
                        detection_method="depth_filtered_out",
                        detection_time_ms=(time.time() - start_time) * 1000
                    )
                    
            except (IndexError, ValueError) as e:
                self.logger.warning(f"깊이 정보 처리 실패: {e}")
        
        # 깊이 정보가 없거나 처리 실패 시 기본 결과 반환
        return LaserDetectionResult(
            detected=best_candidate.get('confidence', 0) > 0,
            confidence=best_candidate.get('confidence', 0),
            position=best_candidate.get('position', (0, 0)),
            rgb_values=best_candidate.get('rgb_values', (0, 0, 0)),
            brightness=best_candidate.get('brightness', 0),
            detection_method=best_candidate.get('detection_method', 'unknown'),
            detection_time_ms=(time.time() - start_time) * 1000,
            screen_coordinate=best_candidate.get('screen_coordinate', (0.0, 0.0)),
            unity_coordinate=best_candidate.get('unity_coordinate', (0.0, 0.0))
        )
    
    def _pixel_to_world_coordinates(self, x: float, y: float, depth_mm: float) -> Tuple[float, float, float]:
        """픽셀 좌표를 3D 월드 좌표로 변환 (간단한 핀홀 모델)"""
        # Azure Kinect 기본 파라미터 (추후 실제 캘리브레이션으로 교체)
        if self.camera_kwargs.get('use_4k', False):
            fx, fy = 2000.0, 2000.0  # 4K용 추정값
            cx, cy = 1920.0, 1080.0
        else:
            fx, fy = 1000.0, 1000.0  # HD용 추정값
            cx, cy = 960.0, 540.0
        
        # 카메라 좌표계로 변환
        world_x = (x - cx) * depth_mm / fx
        world_y = (y - cy) * depth_mm / fy
        world_z = depth_mm
        
        return (world_x, world_y, world_z)
    
    def detect_chessboard_corners(self, frame: np.ndarray, camera_name: str = "primary") -> Tuple[bool, np.ndarray]:
        """
        체스보드 코너 검출 (에피폴라인 캘리브레이션용)
        
        Args:
            frame: 입력 프레임
            camera_name: 카메라 이름 ("primary" 또는 "secondary")
            
        Returns:
            (detected, corners): 검출 성공 여부와 코너 좌표들
        """
        try:
            if frame is None:
                return False, np.array([])
            
            # 그레이스케일 변환
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 체스보드 코너 검출
            pattern_size = self.calibration_config['chessboard_size']
            found, corners = cv2.findChessboardCorners(
                gray, 
                pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            )
            
            if found and len(corners) >= self.calibration_config['min_corners_detected']:
                # 서브픽셀 정확도로 코너 개선
                criteria = self.calibration_config['subpix_criteria']
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                # 실시간 검출 상태만 업데이트 (캘리브레이션 데이터는 별도 수집)
                if camera_name == "primary":
                    self.calibration_data['primary_detected'] = True
                else:
                    self.calibration_data['secondary_detected'] = True
                
                # 로그 출력 완전 비활성화 (너무 많은 메시지 방지)
                # self.logger.info(f"[SUCCESS] {camera_name} 카메라 체스보드 코너 검출 중: {len(corners_refined)}개")
                return True, corners_refined
            
            else:
                # 검출 실패
                if camera_name == "primary":
                    self.calibration_data['primary_detected'] = False
                else:
                    self.calibration_data['secondary_detected'] = False
                
                return False, np.array([])
                
        except Exception as e:
            self.logger.error(f"{camera_name} 카메라 체스보드 검출 실패: {e}")
            return False, np.array([])
    
    def draw_chessboard_corners(self, frame: np.ndarray, corners: np.ndarray, found: bool, 
                               scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
        """
        체스보드 코너를 프레임에 그리기
        
        Args:
            frame: 디스플레이 프레임
            corners: 검출된 코너들
            found: 검출 성공 여부
            scale_x, scale_y: 디스플레이 스케일링 비율
            
        Returns:
            코너가 그려진 프레임
        """
        try:
            if found and len(corners) > 0:
                # 스케일링 적용된 코너들
                scaled_corners = corners.copy()
                scaled_corners[:, 0, 0] *= scale_x
                scaled_corners[:, 0, 1] *= scale_y
                
                # 체스보드 패턴 그리기
                pattern_size = self.calibration_config['chessboard_size']
                cv2.drawChessboardCorners(frame, pattern_size, scaled_corners, found)
                
                # 코너 개수 표시
                corner_count_text = f"Corners: {len(corners)}/{pattern_size[0] * pattern_size[1]}"
                cv2.putText(frame, corner_count_text, (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 첫 번째 코너에 번호 표시 (방향 확인용)
                if len(scaled_corners) > 0:
                    first_corner = tuple(map(int, scaled_corners[0, 0]))
                    cv2.putText(frame, "1", first_corner, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"체스보드 코너 그리기 실패: {e}")
            return frame
    
    def process_dual_cameras(self) -> Optional[LaserDetectionResult]:
        """듀얼 Azure Kinect 카메라 처리 + 지능형 ROI 업데이트"""
        try:
            # 🧠 지능형 ROI 업데이트 (5프레임마다)
            self.roi_update_counter = getattr(self, 'roi_update_counter', 0) + 1
            if self.roi_update_counter >= self.roi_update_interval:
                self._update_intelligent_roi()
                self.roi_update_counter = 0
            
            # 📌 핵심: 단일 모드 성공 로직을 그대로 2번 실행
            
            # Camera #1: Primary (Screen) - 단일 모드와 동일한 방식
            primary_frame_data = self.camera_manager.capture_frame()
            primary_result = None
            if primary_frame_data:
                # 시각화용 프레임 설정 (중요!)
                self.current_frame = primary_frame_data.color_frame
                self.current_depth = primary_frame_data.depth_frame
                self.current_image = primary_frame_data.color_frame.copy()
                
                # 단일 모드와 100% 동일한 검출 방식 (올바른 깊이 프레임 사용)
                # 카메라별 프레임 차이 버퍼 사용: 호출 순간 전용 프로세서로 스왑
                _prev_processor = self.frame_processor
                self.frame_processor = self.frame_processor_primary
                primary_result = self.detect_laser_with_motion_and_depth(
                    primary_frame_data.color_frame, 
                    primary_frame_data.depth_frame,
                    "screen",
                    core=self.laser_core_screen,
                    use_core_motion=True
                )
                self.frame_processor = _prev_processor
                primary_result.detection_method += "_camera0_primary"
            
            # Camera #2: Secondary (Gun + Screen) - 두 경로 병행  
            secondary_frame_data = self.camera_manager.capture_secondary_frame()
            secondary_result = None
            secondary_screen_result = None
            if secondary_frame_data:
                # 보조 카메라 프레임 설정
                self.secondary_frame = secondary_frame_data.color_frame
                self.secondary_depth = secondary_frame_data.depth_frame
                
                # 단일 모드와 100% 동일한 검출 방식 (카메라별 프레임 차이 버퍼 분리)
                _prev_processor = self.frame_processor
                self.frame_processor = self.frame_processor_secondary
                # 총구 전용 간접광/점광 검출 경로 + BT 융합
                muzzle = self.muzzle_detector.detect(secondary_frame_data.color_frame,
                                                     secondary_frame_data.depth_frame)
                bt = None
                if self.enable_bt and self.bt_worker:
                    bt = self.bt_worker.get_latest_result()
                # 간단 융합 규칙: BT가 있으면 BT 예측 위치 근접 후보에 보너스, 없으면 기존 결과 사용
                if muzzle.detected:
                    pos = muzzle.position
                    conf = muzzle.confidence
                    if bt and bt.estimated_muzzle_3d is not None and bt.wrist_2d and bt.index_tip_2d:
                        # BT 2D 예측 근사: 손목-검지 중점 투영값이 있다면 가중치 보너스
                        try:
                            px = (bt.wrist_2d[0] + bt.index_tip_2d[0]) / 2 if (bt.wrist_2d and bt.index_tip_2d) else None
                            py = (bt.wrist_2d[1] + bt.index_tip_2d[1]) / 2 if (bt.wrist_2d and bt.index_tip_2d) else None
                            if px is not None and py is not None:
                                dist = ((pos[0]-px)**2 + (pos[1]-py)**2) ** 0.5
                                if dist < 50:
                                    conf = min(1.0, conf + 0.1)
                        except Exception:
                            pass
                    secondary_result = LaserDetectionResult(
                        detected=True,
                        confidence=conf,
                        position=pos,
                        rgb_values=(0, 0, 0),
                        brightness=muzzle.brightness,
                        detection_method=("muzzle_detector_bt" if bt else "muzzle_detector"),
                        detection_time_ms=0.0,
                        screen_coordinate=(0, 0),
                        unity_coordinate=(0, 0, 0)
                    )
                else:
                    secondary_result = self.detect_laser_with_motion_and_depth(
                        secondary_frame_data.color_frame,
                        secondary_frame_data.depth_frame,
                        "gun",
                        core=self.laser_core_gun,
                        use_core_motion=True
                    )
                    # 깊이 근접 게이트: 총구는 근거리(예: 0.2~1.2m)에 존재해야 함
                    try:
                        if secondary_result and secondary_result.detected and secondary_frame_data.depth_frame is not None:
                            cx, cy = map(int, secondary_result.position)
                            h, w = secondary_frame_data.depth_frame.shape[:2]
                            if 0 <= cx < w and 0 <= cy < h:
                                dx1, dy1 = max(0, cx - 2), max(0, cy - 2)
                                dx2, dy2 = min(w, cx + 3), min(h, cy + 3)
                                dpatch = secondary_frame_data.depth_frame[dy1:dy2, dx1:dx2].astype(np.float32)
                                valid = dpatch[(dpatch > 0) & (dpatch < 10000)]
                                depth_mm = float(np.median(valid)) if valid.size > 0 else -1.0
                                # MuzzleDetector와 동일한 근접 범위 적용
                                near_mm, far_mm = 200.0, 1200.0
                                if depth_mm <= 0 or not (near_mm <= depth_mm <= far_mm):
                                    # 원거리(벽/스크린) 레이저 스팟 오검출 차단
                                    secondary_result.detected = False
                                    secondary_result.detection_method = "gun_rejected_by_depth"
                                else:
                                    secondary_result.depth_mm = depth_mm
                    except Exception:
                        pass

                # 보조 카메라에서도 스크린 히트점 검출 경로 추가(깊이 미사용)
                try:
                    secondary_screen_result = self.detect_laser_with_motion_and_depth(
                        secondary_frame_data.color_frame,
                        None,
                        "screen",
                        core=self.laser_core_screen,
                        use_core_motion=True
                    )
                except Exception:
                    secondary_screen_result = None
                self.frame_processor = _prev_processor

                # BT 외부 캡처 큐에 공유 프레임 enqueue (가능할 때만)
                try:
                    if self.enable_bt and self.bt_worker and getattr(self.bt_worker, 'external_capture_mode', False):
                        raw_cap = secondary_frame_data.camera_info and None
                        # CameraFrame에 raw_capture 필드가 있는 경우 사용
                        raw_cap = getattr(secondary_frame_data, 'raw_capture', None)
                        if raw_cap is not None:
                            self.bt_worker.external_capture_queue.append(raw_cap)
                except Exception:
                    pass
                secondary_result.detection_method += "_camera1_secondary"
            
            # 통계 업데이트
            self.stats['frames_processed'] += 1
            
            # 결과 통합(단순): 스크린(primary) 우선
            final_result = primary_result if (primary_result and primary_result.detected) else secondary_result
            # 전송: 삼각측량 미사용. 타겟 3D=스크린 평면 교차, 총구 3D=secondary 깊이+BT
            if (primary_result and primary_result.detected) or (secondary_result and secondary_result.detected):
                self.stats['detections'] += 1
                try:
                    target_world = None
                    if primary_result and primary_result.detected:
                        target_world = self._intersect_screen_plane_with_primary_ray(primary_result.position)

                    muzzle_world = None
                    if secondary_result and secondary_result.detected and hasattr(secondary_result, 'position'):
                        muzzle_world = self._compute_muzzle_world_from_secondary(
                            secondary_result.position, getattr(secondary_result, 'depth_mm', None)
                        )

                    if target_world is not None and muzzle_world is not None:
                        # 3D 벡터 검증 로그(요약): 거리, 방향 유효성, 스크린 법선과의 각도
                        try:
                            import numpy as _np
                            vec = _np.array(target_world) - _np.array(muzzle_world)
                            length = float(_np.linalg.norm(vec))
                            angle_deg = None
                            if self.screen_plane is not None:
                                n = _np.array(self.screen_plane['normal'], dtype=_np.float64)
                                vn = _np.dot(vec/ (length + 1e-6), n/ (_np.linalg.norm(n)+1e-6))
                                angle_deg = float(_np.degrees(_np.arccos(max(-1.0, min(1.0, vn)))))
                            self.logger.debug(f"[VECTOR] len={length:.1f}mm, angle_to_normal={angle_deg}")
                        except Exception:
                            pass
                        self._send_vector_from_points_millimeters(
                            muzzle_world, target_world,
                            confidence=(primary_result.confidence if primary_result else 0.5)
                        )
                except Exception as e:
                    self.logger.warning(f"듀얼 모드 Unity 전송 스킵: {e}")
            
            # 보조 카메라 결과 저장 (시각화용)
            self._last_secondary_result = secondary_result
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"듀얼 카메라 처리 실패: {e}")
            return None
    
    def _process_single_camera_pipeline(self, camera_id: int, camera_role: str) -> Optional[LaserDetectionResult]:
        """단일 카메라 파이프라인 (듀얼 모드용)"""
        try:
            # 카메라별 프레임 캡처
            if camera_id == 0:
                # Primary 카메라 (Screen 전용)
                frame_data = self.camera_manager.capture_frame()
                self._primary_frame = frame_data.color_frame if frame_data else None
            else:
                # Secondary 카메라 (Gun 전용)  
                frame_data = self.camera_manager.capture_secondary_frame()
                self._secondary_frame = frame_data.color_frame if frame_data else None
            
            if not frame_data:
                return None
            
            # 캘리브레이션 모드에서는 스킵
            if self.calibration_only_mode or self.show_calibration:
                return LaserDetectionResult(
                    detected=False, confidence=0.0, position=(0, 0),
                    rgb_values=(0, 0, 0), brightness=0, detection_method="calibration_skip",
                    detection_time_ms=0.0, screen_coordinate=(0, 0), unity_coordinate=(0, 0, 0)
                )
            
            # 카메라별 ROI 설정 (역할에 따라 분리)
            if camera_role == "screen":
                # Screen 카메라: 중앙 영역 집중 (80% 영역)
                height, width = frame_data.color_frame.shape[:2]
                screen_roi = (
                    int(width * 0.1),   # x1: 10% 마진
                    int(height * 0.1),  # y1: 10% 마진  
                    int(width * 0.9),   # x2: 90%까지
                    int(height * 0.9)   # y2: 90%까지
                )
                self.laser_core.set_roi("screen", screen_roi)
                
                # Screen: 깊이 프레임 무시 (스크린은 깊이 측정 어려움)
                result = self.detect_laser_with_motion_and_depth(
                    frame_data.color_frame, 
                    None,
                    "screen"
                )
                result.detection_method += f"_camera{camera_id}_screen"
                
            else:
                # Gun 카메라: 좌측 상단 영역 집중 (총구 위치)
                height, width = frame_data.color_frame.shape[:2]
                gun_roi = (
                    0,                      # x1: 좌측부터
                    0,                      # y1: 상단부터
                    int(width * 0.6),       # x2: 60%까지
                    int(height * 0.6)       # y2: 60%까지
                )
                self.laser_core.set_roi("gun", gun_roi)
                
                # Gun: 깊이 프레임 활용 (3D 좌표 필요)
                result = self.detect_laser_with_motion_and_depth(
                    frame_data.color_frame,
                    frame_data.depth_frame,
                    "gun"
                )
                result.detection_method += f"_camera{camera_id}_gun"
            
            return result
            
        except Exception as e:
            self.logger.error(f"단일 카메라 파이프라인 실패 (ID: {camera_id}): {e}")
            return None
    
    def _simple_dual_result_merge(self, primary_result: Optional[LaserDetectionResult], 
                                secondary_result: Optional[LaserDetectionResult]) -> Optional[LaserDetectionResult]:
        """단순한 듀얼 결과 통합 (단일 모드 스타일)"""
        try:
            # 둘 다 검출된 경우: 삼각측량 비사용 → 더 높은 신뢰도 반환
            if (primary_result and primary_result.detected and 
                secondary_result and secondary_result.detected):
                return primary_result if primary_result.confidence >= secondary_result.confidence else secondary_result
            
            # Screen만 검출된 경우 (일반적인 상황)
            elif primary_result and primary_result.detected:
                return primary_result
            
            # Gun만 검출된 경우
            elif secondary_result and secondary_result.detected:
                return secondary_result
            
            # 둘 다 검출 안된 경우
            else:
                return primary_result or LaserDetectionResult(
                    detected=False, confidence=0.0, position=(0, 0),
                    rgb_values=(0, 0, 0), brightness=0, detection_method="none",
                    detection_time_ms=0.0
                )
                
        except Exception as e:
            self.logger.error(f"결과 통합 실패: {e}")
            return primary_result
    
    def _triangulate_3d_point(self, primary_result: LaserDetectionResult, 
                             secondary_result: LaserDetectionResult) -> Optional[LaserDetectionResult]:
        """실제 캘리브레이션 데이터 기반 3D 삼각측량"""
        try:
            # 캘리브레이션 데이터 확인
            if not hasattr(self, 'stereo_calibration') or not self.stereo_calibration:

                # 임시 삼각측량 (fallback)
                p1_x, p1_y = primary_result.position
                p2_x, p2_y = secondary_result.position
                triangulated_x = (p1_x + p2_x) / 2.0
                triangulated_y = (p1_y + p2_y) / 2.0
                triangulated_confidence = (primary_result.confidence + secondary_result.confidence) / 2.0
            else:
                # 정확한 삼각측량 수행
                p1_x, p1_y = primary_result.position
                p2_x, p2_y = secondary_result.position
                
                # 스테레오 비전을 통한 3D 복원
                world_3d = self._stereo_triangulate(
                    (p1_x, p1_y),  # Screen camera point
                    (p2_x, p2_y)   # Gun camera point
                )
                
                if world_3d:
                    # 3D 좌표가 있으면 이를 활용
                    triangulated_x, triangulated_y = p1_x, p1_y  # Screen 기준 유지
                    triangulated_confidence = (primary_result.confidence + secondary_result.confidence) / 2.0 + 0.1  # 보너스
                    print(f"[3D TRIANGULATION] World: {world_3d}")
                else:
                    # 3D 복원 실패시 fallback
                    triangulated_x = (p1_x + p2_x) / 2.0
                    triangulated_y = (p1_y + p2_y) / 2.0
                    triangulated_confidence = (primary_result.confidence + secondary_result.confidence) / 2.0
            
            final_result = LaserDetectionResult(
                detected=True,
                confidence=triangulated_confidence,
                position=(int(triangulated_x), int(triangulated_y)),
                rgb_values=primary_result.rgb_values,
                brightness=(primary_result.brightness + secondary_result.brightness) / 2.0,
                detection_method="dual_kinect_triangulation",
                detection_time_ms=max(primary_result.detection_time_ms, secondary_result.detection_time_ms),
                screen_coordinate=primary_result.screen_coordinate,
                unity_coordinate=primary_result.unity_coordinate
            )
            # 3D 좌표가 계산된 경우 결과 객체에 저장 (후속 Unity 전송을 위해 필요)
            try:
                if 'world_3d' in locals() and world_3d is not None:
                    final_result.world_3d_point = tuple(np.asarray(world_3d).flatten().tolist())
            except Exception:
                pass
            
            # 결과 캐싱 (프레임 스킵 시 재사용)
            self._last_primary_result = final_result
            
            # FPS 측정 완료
            frame_end_time = time.time()
            self.last_frame_time = frame_end_time
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"3D 삼각측량 실패: {e}")
            return None
    
    def _stereo_triangulate(self, point1: tuple, point2: tuple) -> Optional[tuple]:
        """스테레오 캘리브레이션 기반 3D 복원"""
        try:
            import cv2
            
            # 캘리브레이션 데이터 로드 (최신 파일)
            calib_file = "calibration_results/stereo_calibration_2025-08-06T20-40-25.json"
            
            try:
                with open(calib_file, 'r') as f:
                    import json
                    calib_data = json.load(f)
                    
                # 캘리브레이션 매트릭스 추출 (구조 변경 반영)
                camera_matrix_1 = np.array(calib_data['primary_camera_intrinsics']['camera_matrix'])
                camera_matrix_2 = np.array(calib_data['secondary_camera_intrinsics']['camera_matrix'])
                dist_coeffs_1 = np.array(calib_data['primary_camera_intrinsics']['distortion'][0])
                dist_coeffs_2 = np.array(calib_data['secondary_camera_intrinsics']['distortion'][0])
                R = np.array(calib_data['stereo_parameters']['rotation_matrix'])
                T = np.array(calib_data['stereo_parameters']['translation_vector']).reshape(3, 1)
                
                # 정규화된 좌표로 변환
                undistorted_1 = cv2.undistortPoints(
                    np.array([[point1]], dtype=np.float32), 
                    camera_matrix_1, 
                    dist_coeffs_1
                )[0][0]
                
                undistorted_2 = cv2.undistortPoints(
                    np.array([[point2]], dtype=np.float32), 
                    camera_matrix_2, 
                    dist_coeffs_2
                )[0][0]
                
                # 3D 삼각측량
                # P1 = [I|0], P2 = [R|T]
                P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
                P2 = np.hstack([R, T])
                
                # 동차 좌표계에서 3D 점 계산
                points_4d = cv2.triangulatePoints(
                    P1, P2, 
                    undistorted_1.reshape(2, 1), 
                    undistorted_2.reshape(2, 1)
                )
                
                # 정규화
                world_3d = points_4d[:3] / points_4d[3]
                
                print(f"[STEREO] 3D Point: {world_3d.flatten()}")
                return tuple(world_3d.flatten())
                
            except FileNotFoundError:
                print(f"[WARN] 캘리브레이션 파일 없음: {calib_file}")
                return None
            except Exception as e:
                print(f"[ERROR] 캘리브레이션 로드 실패: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"스테레오 삼각측량 실패: {e}")
            return None

    def _compute_muzzle_world_from_secondary(self, pixel_xy: tuple, depth_mm: Optional[float]) -> Optional[tuple]:
        """
        Secondary 카메라의 픽셀과 깊이(mm)를 이용해 Secondary 좌표계의 3D 점을 복원한 뒤
        Primary 좌표계로 변환하여 반환한다.
        """
        try:
            if depth_mm is None or depth_mm <= 0:
                return None
            # 캘리브레이션 데이터 필요
            if not hasattr(self, 'stereo_calibration') or not self.stereo_calibration:
                # 로드 시도
                _ = self._stereo_triangulate((0, 0), (0, 0))  # 내부 로딩 경로 재사용
            import numpy as np
            import cv2
            calib_file = "calibration_results/stereo_calibration_2025-08-06T20-40-25.json"
            with open(calib_file, 'r') as f:
                import json
                calib_data = json.load(f)

            K2 = np.array(calib_data['secondary_camera_intrinsics']['camera_matrix'])
            dist2 = np.array(calib_data['secondary_camera_intrinsics']['distortion'][0])
            R = np.array(calib_data['stereo_parameters']['rotation_matrix'])
            T = np.array(calib_data['stereo_parameters']['translation_vector']).reshape(3, 1)

            # Secondary 픽셀 정규화
            undistorted = cv2.undistortPoints(
                np.array([[pixel_xy]], dtype=np.float32), K2, dist2
            )[0][0]  # (x_n, y_n)

            z = depth_mm  # mm
            x = undistorted[0] * z
            y = undistorted[1] * z
            X2 = np.array([[x], [y], [z]])  # Secondary 좌표계 (mm)

            # Primary 좌표계로 변환: X1 = R^T * (X2 - T)
            X1 = R.T @ (X2 - T)
            return (float(X1[0, 0]), float(X1[1, 0]), float(X1[2, 0]))
        except Exception as e:
            self.logger.warning(f"총구 3D 복원 실패: {e}")
            return None

    def _send_vector_from_points_millimeters(self, origin_mm: tuple, target_mm: tuple, confidence: float = 1.0) -> None:
        """원점/타겟 3D(mm)를 받아 Unity 호환 포맷(m)으로 송신"""
        try:
            vector_3d = {
                "message_type": "laser_3d_vector",
                "detected": True,
                "timestamp": time.time(),
                "origin": {"x": origin_mm[0] / 1000.0, "y": origin_mm[1] / 1000.0, "z": origin_mm[2] / 1000.0},
                "target": {"x": target_mm[0] / 1000.0, "y": target_mm[1] / 1000.0, "z": target_mm[2] / 1000.0},
                "direction": {
                    "x": float((target_mm[0] - origin_mm[0]) / max(1e-6, np.linalg.norm(np.array(target_mm) - np.array(origin_mm)))),
                    "y": float((target_mm[1] - origin_mm[1]) / max(1e-6, np.linalg.norm(np.array(target_mm) - np.array(origin_mm)))),
                    "z": float((target_mm[2] - origin_mm[2]) / max(1e-6, np.linalg.norm(np.array(target_mm) - np.array(origin_mm))))
                },
                "vector_length": float(np.linalg.norm(np.array(target_mm) - np.array(origin_mm))) / 1000.0,
                "confidence": float(confidence)
            }
            self._send_vector_to_unity(vector_3d)
        except Exception as e:
            self.logger.warning(f"원점/타겟 송신 실패: {e}")
    
    def detect_laser_with_motion_and_depth(self, frame: np.ndarray, depth_frame: Optional[np.ndarray], roi_type: str = "screen", core: Optional[LaserDetectorCore] = None, use_core_motion: bool = False) -> LaserDetectionResult:
        """
        Azure Kinect RGB 하이브리드 검출: 움직임 + 밝기 + 깊이 결합 검출 (HSV 완전 제거)
        
        핵심 개선:
        1. RGB 하이브리드 검출 → 밝기 + 움직임 + 깊이 결합 (조명 무관)
        2. 적응형 임계값 학습 → 환경 변화 대응
        3. 프레임 차이 최우선 → 레이저 ON/OFF 구분
        4. 단일/듀얼 동일 로직 → 일관성 보장
        """
        start_time = time.time()
        
        detection_result = LaserDetectionResult(
            detected=False, confidence=0.0, position=(0, 0),
            rgb_values=(0, 0, 0), brightness=0, detection_method="none",
            detection_time_ms=0.0, screen_coordinate=(0, 0), unity_coordinate=(0, 0, 0)
        )
        
        try:
            # 깊이 프레임 선택적 사용 (스크린 모드 대응)
            if depth_frame is None:
                self.logger.debug("깊이 프레임 없음 - RGB 전용 모드로 검출")
            
            # 프레임 차이 기반 검출 (핵심: 단일 카메라 성공 방식)
            motion_mask = None
            if not use_core_motion and self.frame_processor.is_frame_diff_enabled():
                # 1단계: 프레임 버퍼 업데이트
                self.frame_processor.update_frame_buffer(frame)
                
                # 2단계: 움직임 영역 검출 (레이저 ON/OFF 순간 캐치)
                motion_result = self.frame_processor.detect_motion_regions(frame)
                motion_mask = motion_result.motion_mask
            
            # 3단계: 레이저 검출 (RGB 우선, RGBD 보조)
            # 단일 모드와 동일한 방식으로 검출
            # 카메라별 코어 선택 (기본값: 단일 코어)
            detect_core = core if core is not None else self.laser_core
            candidates = detect_core.detect_laser_candidates(
                frame, depth_frame, motion_mask, roi_type
            )
            
            if candidates:
                # 최고 신뢰도 후보 선택
                best_candidate = candidates[0]  # 이미 정렬됨
                
                detection_result.detected = True
                # RGB 하이브리드와 호환성 처리
                if 'position' in best_candidate:
                    detection_result.position = best_candidate['position']
                elif 'center' in best_candidate:
                    detection_result.position = best_candidate['center']
                
                detection_result.confidence = best_candidate['confidence']
                detection_result.brightness = best_candidate['brightness']
                detection_result.detection_method = best_candidate.get('detection_method', 'rgbd')
                
                # 3D 좌표 포함 (RGBD의 핵심 장점)
                if 'world_3d' in best_candidate:
                    detection_result.world_3d_point = best_candidate['world_3d']
                if 'depth_mm' in best_candidate:
                    detection_result.depth_mm = best_candidate['depth_mm']
                
                # RGB 값을 검출 결과에 포함
                detection_result.rgb_values = best_candidate.get('rgb_values', (0, 0, 0))
                
                # 🔬 과학적 레이저 검출 성공 로깅
                position = best_candidate.get('position', best_candidate.get('center', (0, 0)))
                scientific_scores = best_candidate.get('scientific_scores', {})
                is_laser_candidate = best_candidate.get('is_green_laser', False)
                detection_method = best_candidate.get('detection_method', 'unknown')
                
                # 과학적 검출 플래그
                science_flag = "[scientific]" if detection_method == 'scientific_bayesian' else "[basic]"
                laser_flag = "[laser]" if is_laser_candidate else ""
                
                self.logger.debug(
                    f"{science_flag} 검출 성공{laser_flag}: 위치={position}, "
                    f"신뢰도={best_candidate['confidence']:.2f}, "
                    f"깊이={best_candidate.get('depth_mm', 0):.0f}mm, "
                    f"베이지안={scientific_scores.get('bayesian_score', 0):.2f}, "
                    f"물리학적={scientific_scores.get('physics_score', 0):.2f}"
                )
            else:
                # RGBD 검출 실패 시 로깅
                motion_status = "모션 검출됨" if motion_mask is not None else "모션 없음"
                self.logger.debug(f"RGBD 검출 실패: {motion_status}")
            
        except Exception as e:
            self.logger.error(f"RGBD 검출 오류: {e}")
            # 오류 시에도 빈 결과 반환 (fallback 제거)
        
        finally:
            processing_time = (time.time() - start_time) * 1000
            detection_result.detection_time_ms = processing_time
        
        return detection_result
    
    def process_frame(self) -> Optional[LaserDetectionResult]:
        """프레임 처리 메인 루프"""
        try:
            # 듀얼 모드인 경우 별도 처리
            if self.camera_type == "dual_azure_kinect":
                return self.process_dual_cameras()
            
            # 단일 카메라 처리
            frame_data = self.camera_manager.capture_frame()
            if not frame_data:
                return None
            
            self.current_frame = frame_data.color_frame
            self.current_depth = frame_data.depth_frame
            self.stats['frames_processed'] += 1
            
            # 단일 모드: 스크린 폴리곤이 없으면 1회 자동 적용(바닥 반사 차단)
            try:
                if getattr(self.laser_core, 'screen_polygon', None) is None and not getattr(self, '_single_polygon_applied', False):
                    calib_base = (1920, 1080)
                    polygon = self._calculate_screen_polygon_from_calibration(calib_base[0], calib_base[1])
                    if polygon and len(polygon) >= 3:
                        cap_w, cap_h = frame_data.color_frame.shape[1], frame_data.color_frame.shape[0]
                        sx = cap_w / float(calib_base[0])
                        sy = cap_h / float(calib_base[1])
                        scaled_poly = [(int(x * sx), int(y * sy)) for (x, y) in polygon]
                        if hasattr(self.laser_core, 'set_screen_polygon'):
                            self.laser_core.set_screen_polygon(scaled_poly)
                        if hasattr(self.laser_core, 'set_roi_enable'):
                            # 사용자 오버라이드가 없을 때만 활성
                            self.laser_core.set_roi_enable(True, user_override=False)
                        self._single_polygon_applied = True
                        print(f"[ROI] (단일) 스크린 폴리곤 자동 적용: {scaled_poly}")
            except Exception as e:
                print(f"[WARN] 단일 모드 스크린 폴리곤 적용 실패: {e}")

            # RGBD 디버깅을 위한 현재 이미지 업데이트
            self.current_image = frame_data.color_frame.copy()
            # RGB 하이브리드 모드 (HSV 완전 제거)
            
            # 레이저 검출 (깊이 정보 활용) - 캘리브레이션 모드에서는 스킵
            if self.calibration_only_mode or self.show_calibration:
                # 캘리브레이션 모드에서는 빈 결과 반환하여 성능 향상
                detection_result = LaserDetectionResult(
                    detected=False, confidence=0.0, position=(0, 0),
                    rgb_values=(0, 0, 0), brightness=0, detection_method="calibration_skip",
                    detection_time_ms=0.0, screen_coordinate=(0, 0), unity_coordinate=(0, 0, 0)
                )
            else:
                # Azure Kinect 개선: 움직임 + RGB 하이브리드 결합 검출 (HSV 완전 제거)
                detection_result = self.detect_laser_with_motion_and_depth(self.current_frame, self.current_depth, "screen")
            
            if detection_result.detected:
                self.stats['detections'] += 1
                
                # 3D 벡터 생성 및 Unity 전송 (개선된 버전)
                try:
                    # 3D 좌표가 있는 경우 벡터 생성
                    if detection_result.world_3d_point and detection_result.depth_mm:
                        vector_3d = self._create_3d_vector(detection_result)
                        if vector_3d:
                            self._send_vector_to_unity(vector_3d)
                            self.stats['triangulated_points'] += 1
                    
                    # 2D 매핑도 유지 (호환성)
                    screen_coord = self.screen_mapper.map_to_screen(detection_result.position)
                    detection_result.screen_coordinate = screen_coord
                    
                    try:
                        unity_coord = self.screen_mapper.convert_to_unity_coordinates(detection_result.position)
                        detection_result.unity_coordinate = unity_coord
                    except AttributeError:
                        # 메서드가 없으면 픽셀 좌표를 그대로 사용
                        detection_result.unity_coordinate = detection_result.position
                    
                except Exception as e:
                    self.logger.error(f"3D 벡터 생성 실패: {e}")
            
            # 결과 캐싱 (단일 카메라용)
            self._last_primary_result = detection_result
            return detection_result
            
        except Exception as e:
            self.logger.error(f"프레임 처리 실패: {e}")
            return None
    
    def _create_3d_vector(self, detection_result) -> Optional[Dict]:
        """3D 벡터 생성 (단일 카메라용)"""
        try:
            if not detection_result.world_3d_point:
                return None
            
            # 카메라 원점 (Azure Kinect 기준)
            camera_origin = np.array([0.0, 0.0, 0.0])  # 카메라 위치
            
            # 레이저 타겟 3D 좌표
            target_3d = np.array(detection_result.world_3d_point)
            
            # 벡터 계산
            direction_vector = target_3d - camera_origin
            vector_magnitude = np.linalg.norm(direction_vector)
            
            if vector_magnitude > 0:
                direction_normalized = direction_vector / vector_magnitude
            else:
                return None
            
            # Unity 전송용 벡터 데이터 (기존 Unity 스크립트와 호환)
            vector_3d = {
                "message_type": "laser_3d_vector",  # Unity_Simple_Target_Hit_System과 호환
                "detected": True,
                "timestamp": time.time(),
                "origin": {
                    "x": camera_origin[0] / 1000.0,  # mm -> m
                    "y": camera_origin[1] / 1000.0,
                    "z": camera_origin[2] / 1000.0
                },
                "target": {
                    "x": target_3d[0] / 1000.0,  # mm -> m  
                    "y": target_3d[1] / 1000.0,
                    "z": target_3d[2] / 1000.0
                },
                "direction": {
                    "x": float(direction_normalized[0]),
                    "y": float(direction_normalized[1]),
                    "z": float(direction_normalized[2])
                },
                "vector_length": vector_magnitude / 1000.0,  # mm -> m
                "confidence": detection_result.confidence,
                # 추가 디버그 정보
                "depth_mm": detection_result.depth_mm,
                "detection_method": detection_result.detection_method,
                "pixel_position": detection_result.position
            }
            
            return vector_3d
            
        except Exception as e:
            self.logger.error(f"3D 벡터 생성 오류: {e}")
            return None
    
    def _send_vector_to_unity(self, vector_3d: Dict):
        """Unity로 3D 벡터 전송"""
        try:
            # JSON 직렬화 및 UDP 전송
            json_data = json.dumps(vector_3d, separators=(',', ':'))
            message_bytes = json_data.encode('utf-8')
            
            # 전송 포트: ScreenMapper 설정을 우선 사용, 없으면 9997 기본값
            target_port = 9997
            try:
                if hasattr(self, 'screen_mapper') and hasattr(self.screen_mapper, 'unity_port'):
                    target_port = int(self.screen_mapper.unity_port)
            except Exception:
                target_port = 9997

            # UDP 소켓이 있으면 재사용, 없으면 임시 소켓 사용
            if hasattr(self.screen_mapper, 'udp_socket') and self.screen_mapper.udp_socket:
                self.screen_mapper.udp_socket.sendto(message_bytes, ("127.0.0.1", target_port))
                self.logger.debug(f"3D 벡터 Unity 전송 성공: {len(message_bytes)} bytes")
            else:
                # UDP 소켓이 없는 경우 새로 생성
                import socket
                udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                udp_socket.sendto(message_bytes, ("127.0.0.1", target_port))
                udp_socket.close()
                self.logger.debug(f"임시 소켓으로 3D 벡터 전송 완료 (port={target_port})")
                
        except Exception as e:
            self.logger.error(f"Unity 3D 벡터 전송 실패: {e}")

    def _send_primary_pixel_ray(self, primary_result: 'LaserDetectionResult') -> None:
        """
        삼각측량이 실패한 경우에도 Unity에서 정합을 빠르게 확인하기 위한 디버그용 레이 송신.
        - 카메라 원점(0,0,0)에서 Primary 픽셀 방향으로 단위벡터를 만들어 보냅니다.
        - 단위: m, 메시지 타입은 laser_3d_vector 그대로.
        """
        try:
            import numpy as np
            # 카메라 좌표계에서 임의의 전진 방향 사용 (Z+), 길이는 fallback
            camera_origin = np.array([0.0, 0.0, 0.0])
            # 화면 정합 확인 목적이므로 전방 단위벡터 사용
            direction_normalized = np.array([0.0, 0.0, 1.0])
            vector_3d = {
                "message_type": "laser_3d_vector",
                "detected": True,
                "timestamp": time.time(),
                "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
                "target": {"x": 0.0, "y": 0.0, "z": 5.0},  # 임시 5m
                "direction": {
                    "x": float(direction_normalized[0]),
                    "y": float(direction_normalized[1]),
                    "z": float(direction_normalized[2])
                },
                "vector_length": 5.0,
                "confidence": max(0.0, float(primary_result.confidence)),
                "pixel_position": primary_result.position,
                "detection_method": primary_result.detection_method
            }
            self._send_vector_to_unity(vector_3d)
            # 디버그 레이 로그는 소음이 많아 INFO에서 DEBUG로 내림
            self.logger.debug("삼각측량 실패 - Primary 픽셀 기반 디버그 레이 송신")
        except Exception as e:
            self.logger.warning(f"디버그 레이 송신 실패: {e}")
    
    def visualize_frame(self, detection_result: Optional[LaserDetectionResult]) -> np.ndarray:
        """프레임 시각화 (화면 크기 조정)"""
        if self.current_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 원본 프레임을 디스플레이용 크기로 조정 (4K → HD)
        original_frame = self.current_frame.copy()
        display_height, display_width = 720, 1280  # HD 크기로 조정
        
        # 4K에서 HD로 스케일링
        if original_frame.shape[1] > display_width:
            display_frame = cv2.resize(original_frame, (display_width, display_height))
            # 스케일링 비율 계산 (검출 결과 좌표 조정용)
            self.display_scale_x = display_width / original_frame.shape[1]
            self.display_scale_y = display_height / original_frame.shape[0]
        else:
            display_frame = original_frame
            self.display_scale_x = 1.0
            self.display_scale_y = 1.0
        
        # 카메라 정보 표시
        camera_info = self.camera_manager.get_camera_info()
        if self.camera_type == "dual_azure_kinect":
            cv2.putText(display_frame, "Camera #1 (Screen Detection)", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, f"Camera: {camera_info['camera_type']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Azure Kinect 추가 정보
        if 'use_4k' in camera_info:
            mode_text = "4K" if camera_info['use_4k'] else "HD"
            cv2.putText(display_frame, f"Mode: {mode_text}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        if 'secondary_available' in camera_info:
            dual_text = "Dual" if camera_info['secondary_available'] else "Single"
            cv2.putText(display_frame, f"Config: {dual_text}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        
        # 검출 결과 시각화 (좌표 스케일링 적용)
        if detection_result and detection_result.detected:
            x, y = detection_result.position
            confidence = detection_result.confidence
            
            # 디스플레이용 좌표로 변환
            display_x = int(x * self.display_scale_x)
            display_y = int(y * self.display_scale_y)
            
            # 레이저 포인트 표시
            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.circle(display_frame, (display_x, display_y), 10, color, 2)
            
            # 정보 표시
            info_text = f"Conf: {confidence:.2f}"
            cv2.putText(display_frame, info_text, (display_x + 15, display_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 원본 좌표 표시 (디버그용)
            coord_text = f"({x}, {y})"
            cv2.putText(display_frame, coord_text, (display_x + 15, display_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            
            # 검출 방법 표시
            method_text = f"Method: {detection_result.detection_method}"
            cv2.putText(display_frame, method_text, (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 캘리브레이션 모드 처리 (실시간 체스보드 검출)
        if self.show_calibration and self.current_frame is not None:
            # 캘리브레이션 모드에서는 매 프레임 체스보드 검출 (실시간 시각화)
            found, corners = self.detect_chessboard_corners(self.current_frame, "primary")
            if found:
                display_frame = self.draw_chessboard_corners(
                    display_frame, corners, found, 
                    self.display_scale_x, self.display_scale_y
                )
            
            # 캘리브레이션 상태 표시 (개선된 버전)
            primary_status = "OK" if self.calibration_data['primary_detected'] else "X"
            secondary_status = "OK" if self.calibration_data['secondary_detected'] else "X"
            collected = self.calibration_data['image_count']
            required = self.calibration_data['min_images_required']
            
            # 간단하고 깔끔한 상태 표시
            status_text = f"Calib: P:{primary_status} S:{secondary_status} | Images: {collected}/{required}"
            cv2.putText(display_frame, status_text, (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # 수집 진행률 바
            if required > 0:
                bar_width = 200
                bar_height = 10
                progress = min(collected / required, 1.0)
                cv2.rectangle(display_frame, (10, 200), (10 + bar_width, 200 + bar_height), (100, 100, 100), -1)
                cv2.rectangle(display_frame, (10, 200), (10 + int(bar_width * progress), 200 + bar_height), (0, 255, 0), -1)
        
        # 통계 표시
        elapsed = time.time() - self.stats['start_time']
        if elapsed > 0:
            self.stats['fps'] = self.stats['frames_processed'] / elapsed
        
        cv2.putText(display_frame, f"FPS: {self.stats['fps']:.1f}", 
                   (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Detections: {self.stats['detections']}", 
                   (10, display_frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Depth Det: {self.stats['depth_detections']}", 
                   (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 듀얼 모드 통계
        if self.camera_type == "dual_azure_kinect":
            cv2.putText(display_frame, f"3D Points: {self.stats['triangulated_points']}", 
                       (200, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # RGB 모드에서는 깊이 맵 창 제거 (불필요)
        
        # 디버그 모드: 모션 검출 시각화
        if self.show_debug and hasattr(self.frame_processor, 'get_latest_motion_mask'):
            motion_mask = self.frame_processor.get_latest_motion_mask()
            if motion_mask is not None:
                # 모션 마스크를 컬러로 변환
                motion_display = cv2.applyColorMap(motion_mask, cv2.COLORMAP_HOT)
                
                # 모션 통계 정보 표시
                motion_pixels = cv2.countNonZero(motion_mask)
                total_pixels = motion_mask.shape[0] * motion_mask.shape[1]
                motion_ratio = motion_pixels / total_pixels * 100
                
                cv2.putText(motion_display, f"Motion Pixels: {motion_pixels} ({motion_ratio:.1f}%)", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(motion_display, "Hot areas = Motion detected", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # 모션 마스크 창 표시 (초기 위치만 설정)
                cv2.namedWindow("Motion Detection Analysis", cv2.WINDOW_NORMAL)
                if not hasattr(self, '_motion_window_positioned'):
                    cv2.moveWindow("Motion Detection Analysis", 50, 600)
                    cv2.resizeWindow("Motion Detection Analysis", 480, 270)
                    self._motion_window_positioned = True
                cv2.imshow("Motion Detection Analysis", motion_display)
        
        return display_frame
    
    def visualize_secondary_frame(self, secondary_result: Optional[LaserDetectionResult]) -> np.ndarray:
        """보조 카메라 프레임 시각화 (듀얼 모드 전용, 화면 크기 조정)"""
        if self.secondary_frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 원본 프레임을 디스플레이용 크기로 조정 (4K → HD)
        original_frame = self.secondary_frame.copy()
        display_height, display_width = 720, 1280  # HD 크기로 조정
        
        # 4K에서 HD로 스케일링
        if original_frame.shape[1] > display_width:
            display_frame = cv2.resize(original_frame, (display_width, display_height))
            # 스케일링 비율 계산
            secondary_scale_x = display_width / original_frame.shape[1]
            secondary_scale_y = display_height / original_frame.shape[0]
        else:
            display_frame = original_frame
            secondary_scale_x = 1.0
            secondary_scale_y = 1.0
        
        # 보조 카메라 정보 표시
        cv2.putText(display_frame, "Camera #2 (Gun Detection)", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Azure Kinect 추가 정보
        camera_info = self.camera_manager.get_camera_info()
        if 'use_4k' in camera_info:
            mode_text = "4K" if camera_info['use_4k'] else "HD"
            cv2.putText(display_frame, f"Mode: {mode_text}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.putText(display_frame, "Secondary Camera", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
        
        # 검출 결과 시각화 (보조 카메라용, 좌표 스케일링 적용)
        if secondary_result and secondary_result.detected:
            x, y = secondary_result.position
            confidence = secondary_result.confidence
            
            # 디스플레이용 좌표로 변환
            display_x = int(x * secondary_scale_x)
            display_y = int(y * secondary_scale_y)
            
            # 레이저 포인트 표시 (보조 카메라는 다른 색상)
            color = (255, 0, 255) if confidence > 0.7 else (255, 255, 0)
            cv2.circle(display_frame, (display_x, display_y), 10, color, 2)
            
            # 정보 표시
            info_text = f"Gun Conf: {confidence:.2f}"
            cv2.putText(display_frame, info_text, (display_x + 15, display_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            # 원본 좌표 표시 (디버그용)
            coord_text = f"({x}, {y})"
            cv2.putText(display_frame, coord_text, (display_x + 15, display_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # 캘리브레이션 모드 처리 (보조 카메라) - 실시간 검출
        if self.show_calibration and self.secondary_frame is not None:
            # 보조 카메라도 매 프레임 체스보드 검출 (실시간 시각화)
            found, corners = self.detect_chessboard_corners(self.secondary_frame, "secondary")
            if found:
                display_frame = self.draw_chessboard_corners(
                    display_frame, corners, found, 
                    secondary_scale_x, secondary_scale_y
                )
            
            # 캘리브레이션 상태 표시 (개선된 버전 - Secondary)
            primary_status = "OK" if self.calibration_data['primary_detected'] else "X"
            secondary_status = "OK" if self.calibration_data['secondary_detected'] else "X"
            collected = self.calibration_data['image_count']
            required = self.calibration_data['min_images_required']
            
            # 간단하고 깔끔한 상태 표시
            status_text = f"Calib: P:{primary_status} S:{secondary_status} | Images: {collected}/{required}"
            cv2.putText(display_frame, status_text, (10, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # 수집 진행률 바
            if required > 0:
                bar_width = 200
                bar_height = 10
                progress = min(collected / required, 1.0)
                cv2.rectangle(display_frame, (10, 200), (10 + bar_width, 200 + bar_height), (100, 100, 100), -1)
                cv2.rectangle(display_frame, (10, 200), (10 + int(bar_width * progress), 200 + bar_height), (255, 0, 255), -1)
        
        # 보조 카메라 전용 통계 표시
        cv2.putText(display_frame, "Gun Detection Mode", 
                   (10, display_frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # RGB 모드에서는 깊이 맵 창 제거 (불필요)
        
        return display_frame
    
    def handle_keyboard_input(self, key: int):
        """키보드 입력 처리"""
        if key == ord('d') or key == ord('D'):
            self.show_debug = not self.show_debug
            print(f"디버그 모드: {'활성화' if self.show_debug else '비활성화'}")
            if self.show_debug:
                print("  - 모션 마스크 시각화 활성화")
                print("  - 검출 영역 분석 활성화")
                print("  - 프레임 차이 세부 정보 표시")
        
        elif key == ord('z') or key == ord('Z'):
            self.show_depth = not self.show_depth
            print(f"📏 깊이 맵 표시: {'활성화' if self.show_depth else '비활성화'}")
        
        elif key == ord('t') or key == ord('T'):
            if self.camera_type == "dual_azure_kinect":
                self.azure_kinect_config['triangulation_enabled'] = not self.azure_kinect_config['triangulation_enabled']
                print(f"🎯 3D 삼각측량: {'활성화' if self.azure_kinect_config['triangulation_enabled'] else '비활성화'}")
        
        elif key == ord('f') or key == ord('F'):
            self.azure_kinect_config['use_depth_filtering'] = not self.azure_kinect_config['use_depth_filtering']
            print(f"🔍 깊이 필터링: {'활성화' if self.azure_kinect_config['use_depth_filtering'] else '비활성화'}")
        
        elif key == ord('h') or key == ord('H'):
            # Enhanced CHT 토글 (새로운 기능)
            if self.enhanced_laser_core is not None:
                current_cht = self.enhanced_laser_core.enable_cht
                self.enhanced_laser_core.toggle_cht(not current_cht)
                print(f"🔍 Modified CHT: {'활성화' if not current_cht else '비활성화'}")
            else:
                print("ℹ️ Enhanced Detector가 비활성화되어 있습니다")
        
        elif key == ord('e') or key == ord('E'):
            # Enhanced Detector 통계 출력 (새로운 기능)
            if self.enhanced_laser_core is not None:
                stats = self.enhanced_laser_core.get_enhanced_stats()
                print("📊 Enhanced Detector 통계:")
                print(f"   CHT 활성화: {stats['cht_enabled']}")
                print(f"   CHT 검증 횟수: {stats['cht_verifications']}")
                print(f"   CHT 개선 횟수: {stats['cht_improvements']}")
                print(f"   CHT 성공률: {stats['cht_success_rate']:.1f}%")
                print(f"   평균 처리 시간: {stats['avg_processing_time_ms']:.2f}ms")
                if 'cht_avg_time_ms' in stats:
                    print(f"   CHT 평균 시간: {stats['cht_avg_time_ms']:.2f}ms")
                    print(f"   CHT 최대 시간: {stats['cht_max_time_ms']:.2f}ms")
            else:
                print("ℹ️ Enhanced Detector가 비활성화되어 있습니다")
        
        elif key == ord('c') or key == ord('C'):
            self.show_calibration = not self.show_calibration
            print(f"🎯 캘리브레이션 모드: {'활성화' if self.show_calibration else '비활성화'}")
            if self.show_calibration:
                print("📋 체스보드 패턴을 두 카메라에 모두 보여주세요")
                print(f"   패턴 크기: {self.calibration_config['chessboard_size'][0]}x{self.calibration_config['chessboard_size'][1]} 내부 코너")
                print("📸 'S' 키: 캘리브레이션 이미지 수집 (20장 필요)")
                print("🔬 'P' 키: 캘리브레이션 계산 수행")
                print("🚀 성능 최적화: 레이저 검출 비활성화됨 (FPS 향상)")
        
        elif key == ord('s') or key == ord('S'):
            # 캘리브레이션 이미지 수집
            if self.show_calibration and self.camera_type == "dual_azure_kinect":
                self.capture_calibration_image()
        
        elif key == ord('p') or key == ord('P'):
            # 캘리브레이션 계산 수행
            if self.show_calibration and self.camera_type == "dual_azure_kinect":
                self.perform_stereo_calibration()
        
        elif key == ord('l') or key == ord('L'):
            # 로그 레벨 조정
            current_level = self.logger.level
            if current_level == 20:  # INFO
                self.logger.setLevel(30)  # WARNING
                print("📝 로그 레벨: WARNING (INFO 메시지 숨김)")
            elif current_level == 30:  # WARNING
                self.logger.setLevel(40)  # ERROR
                print("📝 로그 레벨: ERROR (경고 메시지도 숨김)")
            else:  # ERROR 이상
                self.logger.setLevel(20)  # INFO
                print("📝 로그 레벨: INFO (모든 메시지 표시)")
        
        elif key == ord('w') or key == ord('W'):
            # RGBD 샘플링 (클릭된 위치의 값 사용)
            if hasattr(self, 'last_clicked_rgbd') and self.last_clicked_rgbd:
                try:
                    sample_data = self.last_clicked_rgbd
                    
                    # RGB 학습 함수 호출
                    position = sample_data['position']
                    rgb_values = sample_data['rgb_values']
                    brightness = sample_data['brightness']
                    depth_mm = sample_data.get('depth_mm')
                    
                    # 새로운 RGB 학습 메서드 사용
                    success = self.laser_core.learn_rgb_sample(position, rgb_values, brightness, depth_mm)
                    
                    if success:
                        print(f"[성공] RGB 학습 완료: 밝기={brightness}, RGB={rgb_values}")
                    else:
                        print(f"[경고] RGB 학습에 실패했습니다")
                    
                    print(f"✅ [W키 샘플링 성공] RGBD 범위가 업데이트되었습니다")
                    print(f"   위치: {sample_data['position']}")
                    print(f"   밝기: {sample_data['brightness']}")
                    print(f"   깊이: {sample_data['depth_mm']}mm")
                except Exception as e:
                    print(f"❌ [W키 샘플링 실패] {e}")
            else:
                print("⚠️ 먼저 화면을 클릭하여 샘플링할 위치를 선택하세요")
        
        elif key == ord('r') or key == ord('R'):
            # 통계 리셋
            self.stats = {
                'frames_processed': 0,
                'detections': 0,
                'depth_detections': 0,
                'triangulated_points': 0,
                'start_time': time.time(),
                'fps': 0.0
            }
            
        elif key == ord('m') or key == ord('M'):
            # 🎯 성능 모니터링 출력
            print("\n" + "="*60)
            print("🎯 실시간 성능 모니터링")
            print("="*60)
            
            perf_stats = self.get_performance_stats()
            
            print("📊 FPS 성능:")
            current_fps = perf_stats.get('current_fps', 0)
            target_fps = perf_stats.get('target_fps', 50)
            achievement = perf_stats.get('fps_achievement', '0%')
            
            print(f"   현재 FPS: {current_fps}")
            print(f"   목표 FPS: {target_fps}")
            print(f"   달성률: {achievement}")
            
            if current_fps >= 45:
                print("   상태: ✅ 우수 (45+ FPS)")
            elif current_fps >= 30:
                print("   상태: ⚠️ 양호 (30+ FPS)")  
            else:
                print("   상태: ❌ 개선필요 (<30 FPS)")
            
            print("\n🎯 ROI 효율성:")
            roi_efficiency = perf_stats.get('roi_efficiency', '100%')
            roi_performance = perf_stats.get('roi_performance', 'unknown')
            pixel_reduction = perf_stats.get('pixel_reduction', '0%')
            memory_saved = perf_stats.get('memory_saved', '0%')
            
            print(f"   ROI 면적: {roi_efficiency}")
            print(f"   성능: {roi_performance}")
            print(f"   픽셀 절약: {pixel_reduction}")
            print(f"   메모리 절약: {memory_saved}")
            
            print("\n⚡ 최적화 효과:")
            if float(roi_efficiency.replace('%', '')) < 50:
                print("   ✅ ROI 최적화: 우수")
            elif float(roi_efficiency.replace('%', '')) < 80:
                print("   ⚠️ ROI 최적화: 양호")
            else:
                print("   ❌ ROI 최적화: 개선필요")
            
            print(f"\n🚀 권장사항:")
            if current_fps < 45:
                print("   • ROI 영역 더 축소 권장")
                print("   • 불필요한 검출 옵션 비활성화")
            if float(roi_efficiency.replace('%', '')) > 60:
                print("   • Body Tracking 정확도 개선 필요")
                print("   • 동적 ROI 업데이트 주기 조정")
            
            print("="*60)
            # 캘리브레이션 데이터도 리셋
            self.calibration_data = {
                'primary_corners': [],      # 1번 카메라 코너 좌표들 (각 이미지별)
                'secondary_corners': [],    # 2번 카메라 코너 좌표들 (각 이미지별)
                'object_points': [],        # 3D 월드 좌표 (각 이미지별)
                'primary_images': [],       # 1번 카메라 원본 이미지들
                'secondary_images': [],     # 2번 카메라 원본 이미지들
                'image_count': 0,
                'calibration_complete': False,
                'min_images_required': 15,  # 최소 필요 이미지 수
                'max_images_allowed': 25,   # 최대 수집 이미지 수
                'primary_detected': False,
                'secondary_detected': False,
                'last_capture_time': 0      # 마지막 캡처 시간 (중복 방지)
            }
            print("[INFO] Statistics and Calibration Data Reset Complete")
        
        elif key == ord('o') or key == ord('O'):
            # [quiet] ROI 설정 모드 안내
            print("  0: ROI 비활성화")
        
        elif key == ord('1'):
            # 스크린 ROI 설정 (화면 중앙 영역)
            h, w = self.current_frame.shape[:2] if self.current_frame is not None else (1080, 1920)
            screen_roi = (w//4, h//4, 3*w//4, 3*h//4)  # 중앙 50% 영역
            self.laser_core.set_roi("screen", screen_roi)
            if hasattr(self.laser_core, 'set_roi_enable'):
                self.laser_core.set_roi_enable(True, user_override=True)
            print(f"[quiet] 스크린 ROI 활성화: 중앙 영역 {screen_roi}")
        
        elif key == ord('2'):
            # 총구 ROI 설정 (화면 우측 영역)
            h, w = self.current_frame.shape[:2] if self.current_frame is not None else (1080, 1920)
            gun_roi = (2*w//3, h//4, w, 3*h//4)  # 우측 1/3 영역
            self.laser_core.set_roi("gun", gun_roi)
            print(f"[quiet] 총구 ROI 활성화: 우측 영역 {gun_roi}")
        
        elif key == ord('0'):
            # ROI 비활성화
            if hasattr(self.laser_core, 'set_roi_enable'):
                self.laser_core.set_roi_enable(False, user_override=True)
            else:
                self.laser_core.enable_roi = False
            print("ROI 비활성화(사용자 우선권 고정)")
        
        elif key == ord('=') or key == ord('+'):
            # 밝기 임계값 증가
            self.laser_core.brightness_threshold += 10
            print(f"밝기 임계값 증가: {self.laser_core.brightness_threshold}")
        
        elif key == ord('-') or key == ord('_'):
            # 밝기 임계값 감소
            self.laser_core.brightness_threshold = max(10, self.laser_core.brightness_threshold - 10)
            print(f"밝기 임계값 감소: {self.laser_core.brightness_threshold}")
        
        elif key == ord('['):
            # 신뢰도 임계값 감소 (더 관대하게)
            self.laser_core.min_confidence_threshold = max(0.1, self.laser_core.min_confidence_threshold - 0.1)
            print(f"신뢰도 임계값 감소: {self.laser_core.min_confidence_threshold:.1f}")
        
        elif key == ord(']'):
            # 신뢰도 임계값 증가 (더 엄격하게)
            self.laser_core.min_confidence_threshold = min(1.0, self.laser_core.min_confidence_threshold + 0.1)
            print(f"신뢰도 임계값 증가: {self.laser_core.min_confidence_threshold:.1f}")
        
        elif key == ord(',') or key == ord('<'):
            # 최소 면적 감소 (더 작은 레이저 포인트 허용)
            self.laser_core.min_laser_area = max(1, self.laser_core.min_laser_area - 1)
            print(f"최소 레이저 면적 감소: {self.laser_core.min_laser_area}픽셀")
        
        elif key == ord('.') or key == ord('>'):
            # 최소 면적 증가
            self.laser_core.min_laser_area = min(self.laser_core.max_laser_area - 1, self.laser_core.min_laser_area + 1)
            print(f"최소 레이저 면적 증가: {self.laser_core.min_laser_area}픽셀")
    
    def run_detection(self):
        """메인 검출 루프 실행"""
        if not self.initialize_system():
            print("[ERROR] System Initialization Failed")
            return
        
        print("\n[INFO] Azure Kinect Laser Detection Started!")
        print("[INFO] Keyboard Commands:")
        print("  D: 디버그 모드")
        print("  Z: 깊이 맵 표시")
        print("  T: 3D 삼각측량 (듀얼 모드)")
        print("  F: 깊이 필터링")
        print("  H: Modified CHT 토글 (Enhanced Mode)")
        print("  E: Enhanced Detector 통계 출력")
        print("  C: 캘리브레이션 모드 (체스보드 코너 검출)")
        print("  S: 캘리브레이션 이미지 수집 (C 모드에서)")
        print("  P: 캘리브레이션 계산 및 저장 (C 모드에서)")
        print("  L: 로그 레벨 조정 (INFO → WARNING → ERROR)")
        print("  W: RGBD 샘플링 (클릭 후 W키로 학습)")
        print("  R: 통계 리셋")
        print("  O: ROI 설정 모드")
        print("  1/2/0: ROI 영역 선택/비활성화")
        print("  +/-: 밝기 임계값 조정")
        print("  [/]: 신뢰도 임계값 조정")
        print("  ,/.: 최소 레이저 면적 조정")
        print("  🎯 M: 성능 모니터링 (FPS, ROI 효율성)")
        print("  Q: 종료")
        print("\n[RGBD DEBUG] 마우스 조작:")
        print("  왼쪽 클릭: RGBD 값 확인")
        print("  W키: 클릭한 위치의 RGBD 값 샘플링")
        print("  오른쪽 클릭: 현재 검출 설정 출력")
        
        self.is_running = True
        self.mouse_callback_set = False  # 마우스 콜백 설정 여부 플래그
        
        try:
            while self.is_running:
                try:
                    # 프레임 처리 (예외 처리 강화)
                    detection_result = self.process_frame()
                    
                    # 프레임 처리 실패 시 계속 진행
                    if detection_result is None:
                        print("[WARN] Frame Processing Failed - Continuing...")
                        time.sleep(0.01)  # 짧은 대기
                        continue
                    
                    # 기본 카메라 시각화
                    display_frame = self.visualize_frame(detection_result)
                    
                    if self.camera_type == "dual_azure_kinect":
                        # 듀얼 모드: 두 개의 창 표시 (크기 최적화)
                        # 주 카메라 창 (화면 크기의 1/2로 조정, 사용자가 이동 가능)
                        display_frame_resized = cv2.resize(display_frame, (960, 540))  # 1920x1080 → 960x540
                        cv2.namedWindow("Camera #1 - Screen Detection (Azure Kinect)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow("Camera #1 - Screen Detection (Azure Kinect)", 960, 540)
                        # 초기 위치만 설정, 이후 사용자가 자유롭게 이동 가능
                        if not hasattr(self, '_window1_positioned'):
                            cv2.moveWindow("Camera #1 - Screen Detection (Azure Kinect)", 50, 50)
                            self._window1_positioned = True
                        cv2.imshow("Camera #1 - Screen Detection (Azure Kinect)", display_frame_resized)
                        
                        # 마우스 콜백 설정 (윈도우 생성 후 한 번만)
                        if not self.mouse_callback_set:
                            cv2.setMouseCallback("Camera #1 - Screen Detection (Azure Kinect)", self.mouse_callback)
                            self.mouse_callback_set = True
                            print("[DEBUG] 마우스 콜백 설정 완료")
                        
                        # 보조 카메라 시각화 (보조 카메라 검출 결과 필요)
                        secondary_result = None
                        if hasattr(self, '_last_secondary_result'):
                            secondary_result = self._last_secondary_result
                        
                        secondary_display = self.visualize_secondary_frame(secondary_result)
                        # 보조 카메라 창도 크기 조정 (사용자가 이동 가능)
                        secondary_display_resized = cv2.resize(secondary_display, (960, 540))
                        cv2.namedWindow("Camera #2 - Gun Detection (Azure Kinect)", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                        cv2.resizeWindow("Camera #2 - Gun Detection (Azure Kinect)", 960, 540)
                        # 초기 위치만 설정, 이후 사용자가 자유롭게 이동 가능
                        if not hasattr(self, '_window2_positioned'):
                            cv2.moveWindow("Camera #2 - Gun Detection (Azure Kinect)", 1050, 50)
                            self._window2_positioned = True
                        cv2.imshow("Camera #2 - Gun Detection (Azure Kinect)", secondary_display_resized)
                    else:
                        # 단일 모드: 하나의 창만 표시
                        cv2.imshow("Azure Kinect RGB Laser Detection", display_frame)
                        
                        # 마우스 콜백 설정 (윈도우 생성 후 한 번만)
                        if not self.mouse_callback_set:
                            cv2.setMouseCallback("Azure Kinect RGB Laser Detection", self.mouse_callback)
                            self.mouse_callback_set = True
                            print("[DEBUG] 마우스 콜백 설정 완료")
                    
                    # 키 입력 처리
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        print("[INFO] Normal Exit with Q Key")
                        break
                    elif key != 255:
                        self.handle_keyboard_input(key)
                        
                except Exception as frame_error:
                    print(f"[WARN] Frame Processing Error (Continuing): {frame_error}")
                    time.sleep(0.1)  # 오류 발생 시 잠시 대기
                    continue
        
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by User")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        print("[INFO] System Cleanup...")
        
        self.is_running = False
        
        if self.camera_manager:
            self.camera_manager.cleanup()
        
        cv2.destroyAllWindows()
        
        # 최종 통계 출력
        elapsed = time.time() - self.stats['start_time']
        print(f"\n[INFO] Final Statistics:")
        print(f"  실행시간: {elapsed:.1f}초")
        print(f"  처리 프레임: {self.stats['frames_processed']}")
        print(f"  총 검출: {self.stats['detections']}")
        print(f"  깊이 기반 검출: {self.stats['depth_detections']}")
        if self.camera_type == "dual_azure_kinect":
            print(f"  3D 삼각측량: {self.stats['triangulated_points']}")
        if elapsed > 0:
            print(f"  평균 FPS: {self.stats['frames_processed']/elapsed:.1f}")
        
        print("[INFO] Cleanup Complete")
    
    def capture_calibration_image(self):
        """캘리브레이션을 위한 이미지 수집"""
        try:
            # 현재 시간 확인 (중복 캡처 방지)
            current_time = time.time()
            if current_time - self.calibration_data['last_capture_time'] < 5.0:  # 5초 간격으로 증가
                remaining = 5.0 - (current_time - self.calibration_data['last_capture_time'])
                print(f"⏰ 캡처 간격이 너무 짧습니다. {remaining:.1f}초 후에 다시 시도하세요.")
                return
            
            # 최대 이미지 수 확인
            if self.calibration_data['image_count'] >= self.calibration_data['max_images_allowed']:
                print(f"📸 최대 이미지 수({self.calibration_data['max_images_allowed']})에 도달했습니다.")
                return
            
            # 현재 프레임에서 체스보드 검출
            primary_found, primary_corners = self.detect_chessboard_corners(self.current_frame, "primary")
            secondary_found, secondary_corners = self.detect_chessboard_corners(self.secondary_frame, "secondary")
            
            # 프레임 동기화 검증 (선택적)
            if hasattr(self, 'frame_timestamp_diff') and abs(self.frame_timestamp_diff) > 0.1:  # 100ms 이상 차이
                print(f"⚠️ 프레임 동기화 경고: 카메라 간 시간차 {self.frame_timestamp_diff*1000:.1f}ms")
            
            # 양쪽 카메라에서 모두 검출되어야 함
            if primary_found and secondary_found:
                # 이미지 품질 검증 (중요!)
                primary_quality_ok, primary_quality_msg = self._validate_image_quality(self.current_frame, primary_corners, "primary")
                secondary_quality_ok, secondary_quality_msg = self._validate_image_quality(self.secondary_frame, secondary_corners, "secondary")
                
                if not primary_quality_ok:
                    print(f"⚠️ 1번 카메라 품질 검증 실패: {primary_quality_msg}")
                    return
                    
                if not secondary_quality_ok:
                    print(f"⚠️ 2번 카메라 품질 검증 실패: {secondary_quality_msg}")
                    return
                
                # 3D 월드 좌표 생성 (체스보드 패턴)
                object_points = self._generate_object_points()
                
                # 데이터 저장 (타입 안전성 확보)
                if not isinstance(self.calibration_data['primary_corners'], list):
                    self.calibration_data['primary_corners'] = []
                if not isinstance(self.calibration_data['secondary_corners'], list):
                    self.calibration_data['secondary_corners'] = []
                if not isinstance(self.calibration_data['object_points'], list):
                    self.calibration_data['object_points'] = []
                
                self.calibration_data['primary_corners'].append(primary_corners)
                self.calibration_data['secondary_corners'].append(secondary_corners)
                self.calibration_data['object_points'].append(object_points)
                self.calibration_data['primary_images'].append(self.current_frame.copy())
                self.calibration_data['secondary_images'].append(self.secondary_frame.copy())
                self.calibration_data['image_count'] += 1
                self.calibration_data['last_capture_time'] = current_time
                
                print(f"📸 캘리브레이션 이미지 수집 성공! ({self.calibration_data['image_count']}/{self.calibration_data['min_images_required']})")
                print(f"✅ 품질: {primary_quality_msg}, {secondary_quality_msg}")
                
                # 포즈 다양성 가이드 제공
                remaining = self.calibration_data['min_images_required'] - self.calibration_data['image_count']
                if remaining > 0:
                    poses = ["가까이", "멀리", "왼쪽 기울기", "오른쪽 기울기", "위쪽", "아래쪽", "대각선"]
                    current_pose_idx = (self.calibration_data['image_count'] - 1) % len(poses)
                    next_pose = poses[current_pose_idx] if current_pose_idx < len(poses) else "다양한 각도"
                    print(f"💡 다음 포즈 제안: 체스보드를 '{next_pose}'에서 촬영하세요")
                
                # 이미지 저장 (선택사항)
                import os
                calib_dir = "calibration_images"
                if not os.path.exists(calib_dir):
                    os.makedirs(calib_dir)
                
                cv2.imwrite(f"{calib_dir}/primary_{self.calibration_data['image_count']:02d}.jpg", self.current_frame)
                cv2.imwrite(f"{calib_dir}/secondary_{self.calibration_data['image_count']:02d}.jpg", self.secondary_frame)
                
                # 캘리브레이션 실행 가능 여부 안내
                if self.calibration_data['image_count'] >= self.calibration_data['min_images_required']:
                    print(f"🔬 충분한 이미지가 수집되었습니다. 'P' 키를 눌러 캘리브레이션을 계산하세요.")
            else:
                missing = []
                if not primary_found:
                    missing.append("1번 카메라")
                if not secondary_found:
                    missing.append("2번 카메라")
                print(f"❌ 체스보드 검출 실패: {', '.join(missing)}에서 패턴을 찾을 수 없습니다.")
                
        except Exception as e:
            print(f"❌ 캘리브레이션 이미지 수집 실패: {e}")
    
    def _generate_object_points(self):
        """체스보드 3D 월드 좌표 생성"""
        board_width, board_height = self.calibration_config['chessboard_size']
        square_size = self.calibration_config['square_size']
        
        object_points = np.zeros((board_width * board_height, 3), np.float32)
        object_points[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)
        object_points *= square_size
        
        return object_points
    
    def _validate_image_quality(self, frame: np.ndarray, corners: np.ndarray, camera_name: str) -> Tuple[bool, str]:
        """이미지 품질 검증 (블러, 기울기, 코너 정확도)"""
        try:
            # 1. 이미지 선명도 검사 (Laplacian variance)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            min_sharpness = 100.0  # 최소 선명도 임계값
            
            if laplacian_var < min_sharpness:
                return False, f"이미지가 흐릿함 (선명도: {laplacian_var:.1f} < {min_sharpness})"
            
            # 2. 체스보드 크기 검증 (너무 작거나 큰 경우 제외)
            if len(corners) > 0:
                # 코너들의 바운딩 박스 계산
                x_coords = corners[:, 0, 0]
                y_coords = corners[:, 0, 1]
                width = np.max(x_coords) - np.min(x_coords)
                height = np.max(y_coords) - np.min(y_coords)
                
                # 이미지 대비 체스보드 크기 비율
                img_area = frame.shape[0] * frame.shape[1]
                board_area = width * height
                area_ratio = board_area / img_area
                
                # 적절한 크기 범위 확인 (이미지의 1-70%) - 완화된 기준
                if area_ratio < 0.01:
                    return False, f"체스보드가 너무 작음 (면적비: {area_ratio:.1%})"
                if area_ratio > 0.7:
                    return False, f"체스보드가 너무 큼 (면적비: {area_ratio:.1%})"
            
            # 3. 코너 분포 균등성 검증
            if len(corners) > 4:
                # 코너들이 이미지 전체에 고르게 분포되어 있는지 확인
                x_std = np.std(corners[:, 0, 0])
                y_std = np.std(corners[:, 0, 1])
                min_spread = 20.0  # 최소 분산 (완화된 기준)
                
                if x_std < min_spread or y_std < min_spread:
                    return False, f"코너 분포가 불균등함 (X편차:{x_std:.1f}, Y편차:{y_std:.1f})"
            
            # 4. 체스보드 기울기 검증 (적당한 기울기는 허용)
            if len(corners) >= 4:
                pattern_size = self.calibration_config['chessboard_size']
                if len(corners) >= pattern_size[0]:
                    # 첫 번째 행의 기울기
                    first_row = corners[:pattern_size[0]]
                    dx = first_row[-1, 0, 0] - first_row[0, 0, 0]
                    dy = first_row[-1, 0, 1] - first_row[0, 0, 1]
                    angle = abs(np.degrees(np.arctan2(dy, dx)))
                    
                    # 너무 심하게 기울어진 경우만 제외 (60도 이상)
                    if angle > 60 and angle < 120:
                        return False, f"체스보드가 너무 기울어짐 (각도: {angle:.1f}°)"
            
            return True, f"품질 검증 통과 (선명도: {laplacian_var:.1f})"
            
        except Exception as e:
            return False, f"품질 검증 실패: {e}"
    
    def perform_stereo_calibration(self):
        """스테레오 캘리브레이션 계산 및 저장"""
        try:
            if self.calibration_data['image_count'] < self.calibration_data['min_images_required']:
                print(f"❌ 캘리브레이션을 위한 최소 이미지 수가 부족합니다. ({self.calibration_data['image_count']}/{self.calibration_data['min_images_required']})")
                return
            
            print("🔬 스테레오 캘리브레이션 계산 시작...")
            
            # 이미지 크기 가져오기
            image_size = (self.current_frame.shape[1], self.current_frame.shape[0])
            
            # 개별 카메라 캘리브레이션
            print("📹 1번 카메라 캘리브레이션...")
            ret1, camera_matrix1, distortion1, rvecs1, tvecs1 = cv2.calibrateCamera(
                self.calibration_data['object_points'],
                self.calibration_data['primary_corners'],
                image_size,
                None, None
            )
            
            print("📹 2번 카메라 캘리브레이션...")
            ret2, camera_matrix2, distortion2, rvecs2, tvecs2 = cv2.calibrateCamera(
                self.calibration_data['object_points'],
                self.calibration_data['secondary_corners'],
                image_size,
                None, None
            )
            
            # 스테레오 캘리브레이션
            print("🎯 스테레오 캘리브레이션...")
            ret, camera_matrix1, distortion1, camera_matrix2, distortion2, R, T, E, F = cv2.stereoCalibrate(
                self.calibration_data['object_points'],
                self.calibration_data['primary_corners'],
                self.calibration_data['secondary_corners'],
                camera_matrix1, distortion1,
                camera_matrix2, distortion2,
                image_size,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
                flags=cv2.CALIB_FIX_INTRINSIC
            )
            
            # 스테레오 정류화
            print("📐 스테레오 정류화 계산...")
            R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
                camera_matrix1, distortion1,
                camera_matrix2, distortion2,
                image_size, R, T,
                flags=cv2.CALIB_ZERO_DISPARITY,
                alpha=0.9
            )
            
            # 정류화 맵 생성
            map1x, map1y = cv2.initUndistortRectifyMap(camera_matrix1, distortion1, R1, P1, image_size, cv2.CV_32FC1)
            map2x, map2y = cv2.initUndistortRectifyMap(camera_matrix2, distortion2, R2, P2, image_size, cv2.CV_32FC1)
            
            # 결과 저장 (ZED 버전과 유사한 상세 구조)
            baseline_mm = float(np.linalg.norm(T))
            convergence_angle = float(np.degrees(np.arccos(np.clip(R[0, 0], -1, 1))))
            
            self.calibration_results = {
                'calibration_info': {
                    'method': 'Azure Kinect Dual Camera Stereo Calibration',
                    'calibration_date': time.strftime('%Y-%m-%dT%H:%M:%S'),
                    'num_images': self.calibration_data['image_count'],
                    'chessboard_size': self.calibration_config['chessboard_size'],
                    'square_size_mm': self.calibration_config['square_size'],
                    'image_resolution': list(image_size),
                    'camera_type': 'Azure Kinect DK'
                },
                'primary_camera_intrinsics': {
                    'camera_matrix': camera_matrix1.tolist(),
                    'distortion': distortion1.tolist(),
                    'rms_error': float(ret1),
                    'fx_fy_ratio': float(camera_matrix1[0, 0] / camera_matrix1[1, 1]),
                    'focal_length_x': float(camera_matrix1[0, 0]),
                    'focal_length_y': float(camera_matrix1[1, 1]),
                    'principal_point_x': float(camera_matrix1[0, 2]),
                    'principal_point_y': float(camera_matrix1[1, 2])
                },
                'secondary_camera_intrinsics': {
                    'camera_matrix': camera_matrix2.tolist(),
                    'distortion': distortion2.tolist(),
                    'rms_error': float(ret2),
                    'fx_fy_ratio': float(camera_matrix2[0, 0] / camera_matrix2[1, 1]),
                    'focal_length_x': float(camera_matrix2[0, 0]),
                    'focal_length_y': float(camera_matrix2[1, 1]),
                    'principal_point_x': float(camera_matrix2[0, 2]),
                    'principal_point_y': float(camera_matrix2[1, 2])
                },
                'stereo_parameters': {
                    'rotation_matrix': R.tolist(),
                    'translation_vector': T.tolist(),
                    'essential_matrix': E.tolist(),
                    'fundamental_matrix': F.tolist(),
                    'stereo_rms_error': float(ret),
                    'baseline_mm': baseline_mm,
                    'rectification_rotation_1': R1.tolist(),
                    'rectification_rotation_2': R2.tolist(),
                    'rectification_projection_1': P1.tolist(),
                    'rectification_projection_2': P2.tolist(),
                    'disparity_to_depth_matrix': Q.tolist()
                },
                'coordinate_system': {
                    'origin': 'Primary camera center',
                    'x_axis': 'Right direction of primary camera',
                    'y_axis': 'Down direction of primary camera', 
                    'z_axis': 'Forward direction of primary camera',
                    'units': 'millimeters',
                    'note': 'Secondary camera position relative to primary camera'
                },
                'quality_metrics': {
                    'overall_stereo_rms_error': float(ret),
                    'primary_camera_rms_error': float(ret1),
                    'secondary_camera_rms_error': float(ret2),
                    'baseline_distance_mm': baseline_mm,
                    'convergence_angle_degrees': convergence_angle,
                    'images_used': self.calibration_data['image_count'],
                    'total_corner_points': self.calibration_data['image_count'] * (self.calibration_config['chessboard_size'][0] * self.calibration_config['chessboard_size'][1])
                }
            }
            
            # JSON 파일로 저장 (폴더 생성 후 저장)
            import json
            import os
            
            # 캘리브레이션 결과 폴더 생성
            calibration_folder = "calibration_results"
            os.makedirs(calibration_folder, exist_ok=True)
            
            # 파일명과 전체 경로 생성
            date_str = self.calibration_results['calibration_info']['calibration_date'].replace(':', '-')
            filename = f"stereo_calibration_{date_str}.json"
            full_path = os.path.join(calibration_folder, filename)
            absolute_path = os.path.abspath(full_path)
            
            with open(full_path, 'w', encoding='utf-8') as f:
                json.dump(self.calibration_results, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 스테레오 캘리브레이션 완료!")
            print(f"📁 저장 폴더: {calibration_folder}/")
            print(f"📄 결과 파일: {filename}")
            print(f"🗂️ 전체 경로: {absolute_path}")
            
            # 상세한 품질 메트릭스 출력
            print(f"\n📊 캘리브레이션 품질 분석:")
            print(f"  전체 스테레오 RMS 오차: {ret:.4f} 픽셀")
            print(f"  1번 카메라 RMS 오차: {ret1:.4f} 픽셀")
            print(f"  2번 카메라 RMS 오차: {ret2:.4f} 픽셀")
            print(f"  베이스라인 거리: {baseline_mm:.2f} mm")
            print(f"  수렴각도: {convergence_angle:.2f}°")
            print(f"  사용된 이미지: {self.calibration_data['image_count']}장")
            print(f"  총 코너 포인트: {self.calibration_data['image_count'] * (self.calibration_config['chessboard_size'][0] * self.calibration_config['chessboard_size'][1])}개")
            
            # 품질 평가
            if ret < 0.5:
                print(f"🟢 우수한 캘리브레이션 품질!")
            elif ret < 1.0:
                print(f"🟡 양호한 캘리브레이션 품질")
            elif ret < 2.0:
                print(f"🟠 허용 가능한 캘리브레이션 품질")
            else:
                print(f"🔴 캘리브레이션 품질 개선 필요 (권장: <2.0 픽셀)")
            
            self.calibration_data['calibration_complete'] = True
            
        except Exception as e:
            print(f"❌ 스테레오 캘리브레이션 실패: {e}")
            import traceback
            traceback.print_exc()
    
    def mouse_callback(self, event, x, y, flags, param):
        """마우스 콜백 함수 - RGBD 실시간 디버깅
        - 컬러/깊이 해상도 불일치 시 좌표 스케일 매핑 적용
        - 깊이 0(미측정)일 때 주변 탐색으로 보정
        """
        if self.current_image is None:
            return

        # 좌표 및 프레임 크기 확보
        color_h, color_w = self.current_image.shape[:2]
        depth_available = self.current_depth is not None
        if depth_available:
            depth_h, depth_w = self.current_depth.shape[:2]
        else:
            depth_h = depth_w = 0

        # 컬러 좌표 → 깊이 좌표 스케일링 (정렬 미보장 환경에서 근사 매핑)
        def color_to_depth_coords(cx: int, cy: int) -> tuple:
            if not depth_available or color_w <= 0 or color_h <= 0:
                return (-1, -1)
            sx = depth_w / float(color_w)
            sy = depth_h / float(color_h)
            dx = int(round(cx * sx))
            dy = int(round(cy * sy))
            # 경계 클리핑
            dx = max(0, min(depth_w - 1, dx)) if depth_w > 0 else -1
            dy = max(0, min(depth_h - 1, dy)) if depth_h > 0 else -1
            return (dx, dy)

        if event == cv2.EVENT_LBUTTONDOWN:
            # 왼쪽 클릭: RGBD 값 확인 및 샘플링
            if (0 <= y < color_h and 0 <= x < color_w):
                # RGB 값 추출
                bgr_pixel = self.current_image[y, x]
                b, g, r = bgr_pixel
                brightness = int(cv2.cvtColor(self.current_image[y:y+1, x:x+1], cv2.COLOR_BGR2GRAY)[0, 0])

                # 깊이 값 추출 (가능한 경우)
                depth_mm = None
                if depth_available:
                    dx, dy = color_to_depth_coords(x, y)
                    if dx >= 0 and dy >= 0:
                        raw_depth = self.current_depth[dy, dx]
                        # Azure Kinect 깊이 값 유효성 검사
                        if 0 < raw_depth < 65535:
                            depth_mm = float(raw_depth)
                        else:
                            # 주변 픽셀에서 유효한 깊이 값 찾기 (5x5 영역으로 확장)
                            found = False
                            for ry in range(-2, 3):
                                for rx in range(-2, 3):
                                    ny, nx = dy + ry, dx + rx
                                    if 0 <= ny < depth_h and 0 <= nx < depth_w:
                                        neighbor_depth = self.current_depth[ny, nx]
                                        if 0 < neighbor_depth < 65535:
                                            depth_mm = float(neighbor_depth)
                                            found = True
                                            break
                                if found:
                                    break
                
                print(f"\n[RGBD DEBUG] 클릭 위치 ({x}, {y}):")
                print(f"  RGB 값: R={r}, G={g}, B={b}")
                print(f"  밝기 값: {brightness}")
                if depth_mm is not None:
                    print(f"  깊이 값: {depth_mm}mm ({depth_mm/1000:.2f}m)")
                else:
                    print(f"  깊이 값: 없음")
                    print(f"  [참고] 화면/광택/투명 표면은 깊이 측정이 어려움 (정렬 미보정 상태에서는 근사 오차 가능)")
                
                # 현재 RGBD 검출 범위와 비교
                actual_brightness_threshold = self.laser_core.brightness_threshold
                actual_depth_range = self.laser_core.depth_range_mm
                
                brightness_match = brightness >= actual_brightness_threshold
                depth_match = (depth_mm is None or 
                             (actual_depth_range[0] <= depth_mm <= actual_depth_range[1]))
                
                print(f"  실제 밝기 임계값: {actual_brightness_threshold} ({'통과' if brightness_match else '실패'})")
                print(f"  실제 깊이 범위: {actual_depth_range[0]}-{actual_depth_range[1]}mm ({'통과' if depth_match else '실패'})")
                print(f"  검출 가능성: {'가능' if brightness_match and depth_match else '불가능'}")
                

                # RGBD 값 저장 (W키 샘플링용)
                self.last_clicked_rgbd = {
                    'position': (x, y),
                    'brightness': brightness,
                    'rgb_values': (r, g, b),
                    'depth_mm': depth_mm if depth_mm is not None else 0,
                    'confidence': 0.8
                }
                print(f"  [팁] W키를 눌러 이 RGBD 값을 샘플링할 수 있습니다")
        
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 오른쪽 클릭: 실제 샘플링된 RGBD 검출 설정 출력
            print(f"\n[실제 검출 설정] 현재 Azure Kinect RGBD 검출 범위:")
            actual_brightness_threshold = self.laser_core.brightness_threshold
            actual_depth_range = self.laser_core.depth_range_mm
            actual_min_area = self.laser_core.min_laser_area
            actual_max_area = self.laser_core.max_laser_area
            
            print(f"  밝기 임계값: {actual_brightness_threshold}")
            print(f"  깊이 범위: {actual_depth_range[0]}-{actual_depth_range[1]}mm")
            print(f"  면적 범위: {actual_min_area}-{actual_max_area}픽셀")
            
            # 샘플링된 데이터 수 확인
            if hasattr(self.laser_core, 'learned_samples'):
                learned_count = len(self.laser_core.learned_samples)
                print(f"  샘플링된 RGB 데이터 수: {learned_count}개 (RGB 하이브리드 모드)")
                if hasattr(self.laser_core, 'adaptive_brightness_threshold'):
                    print(f"  적응형 밝기 임계값: {self.laser_core.adaptive_brightness_threshold}")
            else:
                print(f"  샘플링된 데이터 수: 0개 (초기화 필요)")
            
            if learned_count == 0:
                print(f"  [주의] 아직 샘플링된 데이터가 없습니다. W키로 레이저를 샘플링해주세요.")
    
    def _setup_dual_mode_roi(self):
        """듀얼 모드 지능형 ROI 시스템 설정"""
        try:
            print("[INFO] 🧠 지능형 ROI 시스템 초기화 중...")
            
            # HD 모드 강제 적용 (성능 최적화)
            self.use_4k = False
            width, height = 1920, 1080  # HD 해상도로 고정
            print(f"[PERF] 🚀 HD 모드 적용: {width}x{height} @ 30 FPS → 50+ FPS 목표")
            
            # 성능 최적화 설정
            self._apply_performance_optimizations()
            
            # Screen ROI/Polygon: 캘리브레이션 기준(1920x1080)에서 계산 후 캡처 해상도로 스케일
            calib_base = (1920, 1080)
            calib_roi = self._calculate_screen_roi_from_calibration(calib_base[0], calib_base[1])
            camera_info = self.camera_manager.get_camera_info()
            cap_w, cap_h = camera_info.get('resolution', (calib_base[0], calib_base[1]))
            screen_roi = self._scale_roi_to_capture(calib_roi, calib_base, (cap_w, cap_h))
            # 폴리곤 계산 및 적용
            try:
                polygon = self._calculate_screen_polygon_from_calibration(calib_base[0], calib_base[1])
                if polygon and len(polygon) >= 3:
                    sx = cap_w / float(calib_base[0])
                    sy = cap_h / float(calib_base[1])
                    scaled_poly = [(int(x * sx), int(y * sy)) for (x, y) in polygon]
                    if hasattr(self.laser_core, 'set_screen_polygon'):
                        self.laser_core.set_screen_polygon(scaled_poly)
                        print(f"[ROI] 스크린 폴리곤 적용: {scaled_poly}")
            except Exception as e:
                print(f"[WARN] 스크린 폴리곤 계산 실패: {e}")
            
            # Muzzle ROI: Body Tracking 기반 동적 ROI (초기값은 상체 중심)
            initial_muzzle_roi = self._calculate_initial_muzzle_roi(cap_w, cap_h)
            
            # Laser Core에 ROI 설정
            self.laser_core.set_roi("screen", screen_roi)
            self.laser_core.set_roi("gun", initial_muzzle_roi)
            
            # ✅ 지능형 ROI 활성화 (사용자 오버라이드 우선 존중)
            if hasattr(self.laser_core, 'set_roi_enable'):
                self.laser_core.set_roi_enable(True, user_override=False)
            else:
                self.laser_core.enable_roi = True
            print("[SUCCESS] 지능형 ROI 시스템 활성화")
            
            # ROI 업데이트 카운터 초기화
            self.roi_update_counter = 0
            self.roi_update_interval = 5  # 5프레임마다 ROI 업데이트
            
            print(f"[ROI] 스크린 영역: {screen_roi}")
            print(f"[ROI] 총구 영역 (초기): {initial_muzzle_roi}")
            print("[INFO] 🎯 지능형 ROI 시스템 설정 완료 - 50+ FPS 목표")
            
        except Exception as e:
            print(f"[ERROR] 지능형 ROI 설정 실패: {e}")
            # Fallback: 기존 방식으로 동작
            if hasattr(self.laser_core, 'set_roi_enable'):
                self.laser_core.set_roi_enable(False, user_override=False)
            else:
                self.laser_core.enable_roi = False
            print("[FALLBACK] ROI 비활성화로 대체")
    
    def _calculate_screen_roi_from_calibration(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Calibration 데이터 기반 정확한 스크린 ROI 계산"""
        try:
            # calibration 파일에서 screen_plane 데이터 로드
            import json
            with open('azure_kinect_3d_calibration.json', 'r') as f:
                calibration = json.load(f)
            
            # 스크린 3D 코너 좌표들
            corners_3d = calibration['screen_plane']['corners_3d']
            screen_intrinsics = calibration['screen_camera_intrinsics']
            
            # 3D → 2D 투영
            corners_2d = []
            fx, fy = screen_intrinsics['fx'], screen_intrinsics['fy']  
            cx, cy = screen_intrinsics['cx'], screen_intrinsics['cy']
            
            for x3d, y3d, z3d in corners_3d:
                if z3d > 0:  # 카메라 앞쪽에 있는 점만
                    x2d = int(fx * x3d / z3d + cx)
                    y2d = int(fy * y3d / z3d + cy)
                    # 화면 범위 내 클리핑
                    x2d = max(0, min(width-1, x2d))
                    y2d = max(0, min(height-1, y2d))
                    corners_2d.append((x2d, y2d))
            
            if len(corners_2d) >= 3:
                # Bounding Rectangle 계산
                x_coords = [p[0] for p in corners_2d]
                y_coords = [p[1] for p in corners_2d]
                
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                
                # 10% 마진 추가 (투영 오차 보정)
                margin_x = int((x2 - x1) * 0.1)
                margin_y = int((y2 - y1) * 0.1)
                
                roi = (
                    max(0, x1 - margin_x),
                    max(0, y1 - margin_y), 
                    min(width, x2 + margin_x),
                    min(height, y2 + margin_y)
                )
                
                print(f"[CALIBRATION] 스크린 ROI 계산 성공: {roi}")
                return roi
            
        except Exception as e:
            print(f"[WARN] Calibration 기반 ROI 계산 실패: {e}")
        
        # Fallback: 중앙 80% 영역
        margin_x, margin_y = int(width * 0.1), int(height * 0.1)
        fallback_roi = (margin_x, margin_y, width - margin_x, height - margin_y)
        print(f"[FALLBACK] 기본 스크린 ROI 사용: {fallback_roi}")
        return fallback_roi

    def _calculate_screen_polygon_from_calibration(self, width: int, height: int) -> Optional[List[Tuple[int,int]]]:
        """Calibration 데이터 기반 스크린 2D 폴리곤(픽셀) 계산
        - 3D 스크린 코너를 카메라1(스크린 카메라) 내참수로 투영하여 2D 폴리곤을 만든다
        """
        try:
            import json
            with open('azure_kinect_3d_calibration.json', 'r') as f:
                calibration = json.load(f)

            corners_3d = calibration['screen_plane']['corners_3d']
            intr = calibration['screen_camera_intrinsics']
            fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']

            pts: List[Tuple[int,int]] = []
            for x3d, y3d, z3d in corners_3d:
                if z3d > 0:
                    u = int(fx * x3d / z3d + cx)
                    v = int(fy * y3d / z3d + cy)
                    u = max(0, min(width - 1, u))
                    v = max(0, min(height - 1, v))
                    pts.append((u, v))

            if len(pts) >= 3:
                # 시계/반시계 정렬 보장(컨벡스 헐 형식으로 간단 정렬)
                import numpy as np
                pts_np = np.array(pts, dtype=np.int32)
                hull = cv2.convexHull(pts_np)
                hull_list = [(int(p[0][0]), int(p[0][1])) for p in hull]
                return hull_list
        except Exception as e:
            print(f"[WARN] 스크린 폴리곤 계산 실패: {e}")
        return None

    def _scale_roi_to_capture(self, roi: Tuple[int, int, int, int], from_size: Tuple[int, int], to_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """ROI를 기준 해상도(from_size)에서 실제 캡처 해상도(to_size)로 스케일 변환
        
        Args:
            roi: (x1, y1, x2, y2) in from_size pixels
            from_size: (width_from, height_from)
            to_size: (width_to, height_to)
        Returns:
            스케일된 ROI (정수, 화면 경계 내 클리핑)
        """
        try:
            fx = float(to_size[0]) / max(1, from_size[0])
            fy = float(to_size[1]) / max(1, from_size[1])
            x1, y1, x2, y2 = roi
            sx1 = int(round(x1 * fx))
            sy1 = int(round(y1 * fy))
            sx2 = int(round(x2 * fx))
            sy2 = int(round(y2 * fy))
            # 경계 클리핑
            sx1 = max(0, min(to_size[0]-1, sx1))
            sy1 = max(0, min(to_size[1]-1, sy1))
            sx2 = max(0, min(to_size[0], sx2))
            sy2 = max(0, min(to_size[1], sy2))
            return (sx1, sy1, sx2, sy2)
        except Exception:
            return roi
    
    def _calculate_initial_muzzle_roi(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """초기 총구 ROI 설정 (Body Tracking 활성화 전까지 사용)"""
        # 상체 중심 영역 (총구가 주로 나타나는 위치)
        center_x, center_y = width // 2, height // 3  # 화면 중앙 상단
        roi_size = 400  # 400x400 픽셀 영역
        
        x1 = max(0, center_x - roi_size // 2)
        y1 = max(0, center_y - roi_size // 2)
        x2 = min(width, center_x + roi_size // 2)
        y2 = min(height, center_y + roi_size // 2)
        
        roi = (x1, y1, x2, y2)
        print(f"[INITIAL] 총구 ROI (Body Tracking 대기): {roi}")
        return roi
    
    def _update_dynamic_muzzle_roi(self, bt_result, width: int, height: int) -> Tuple[int, int, int, int]:
        """Body Tracking 결과 기반 동적 총구 ROI 업데이트"""
        try:
            if (bt_result and 
                hasattr(bt_result, 'wrist_2d') and bt_result.wrist_2d and
                hasattr(bt_result, 'index_tip_2d') and bt_result.index_tip_2d):
                
                wrist_x, wrist_y = bt_result.wrist_2d
                index_x, index_y = bt_result.index_tip_2d
                
                # 손목 → 손가락 벡터 계산
                vec_x = index_x - wrist_x
                vec_y = index_y - wrist_y
                vec_len = (vec_x**2 + vec_y**2)**0.5
                
                if vec_len > 10:  # 최소 벡터 길이 검증
                    # 벡터 정규화
                    vec_x /= vec_len
                    vec_y /= vec_len
                    
                    # 총기 길이 가정 (600mm → 픽셀 변환)
                    gun_length_px = 150  # 대략적인 픽셀 길이
                    
                    # 총구 위치 예측 (손가락 → 총구 방향으로 연장)
                    muzzle_x = int(index_x + vec_x * gun_length_px)
                    muzzle_y = int(index_y + vec_y * gun_length_px)
                    
                    # ROI 영역 설정 (300x300 픽셀, 충분한 여유)
                    roi_size = 300
                    x1 = max(0, muzzle_x - roi_size // 2)
                    y1 = max(0, muzzle_y - roi_size // 2)
                    x2 = min(width, muzzle_x + roi_size // 2)
                    y2 = min(height, muzzle_y + roi_size // 2)
                    
                    roi = (x1, y1, x2, y2)
                    print(f"[DYNAMIC] 총구 ROI 업데이트: {roi} (예측 위치: {muzzle_x},{muzzle_y})")
                    return roi
            
        except Exception as e:
            print(f"[WARN] 동적 ROI 계산 실패: {e}")
        
        # Fallback: 현재 ROI 유지 또는 초기 ROI
        return self._calculate_initial_muzzle_roi(width, height)
    
    def _update_intelligent_roi(self):
        """지능형 ROI 시스템 업데이트 (5프레임마다 호출)"""
        try:
            # 실제 캡처 해상도를 사용 (성능 및 좌표 일치)
            camera_info = self.camera_manager.get_camera_info()
            cap_w, cap_h = camera_info.get('resolution', (1920, 1080))
            width, height = int(cap_w), int(cap_h)
            
            # Body Tracking 결과 확인
            bt_result = None
            if hasattr(self, 'bt_worker') and self.bt_worker:
                bt_result = self.bt_worker.get_latest_result()
            
            # 동적 총구 ROI 업데이트 (이미 cap_w, cap_h 기반)
            if bt_result:
                new_muzzle_roi = self._update_dynamic_muzzle_roi(bt_result, width, height)
                self.laser_core.set_roi("gun", new_muzzle_roi)
                # [quiet] 총구 ROI 동적 업데이트
            else:
                # Body Tracking 실패 시 더 넓은 영역으로 확장
                fallback_roi = self._calculate_fallback_muzzle_roi(width, height)
                self.laser_core.set_roi("gun", fallback_roi)
                # [quiet] 총구 ROI 폴백 적용
                
            # 성능 모니터링
            if hasattr(self, 'stats'):
                roi_area_ratio = self._calculate_roi_efficiency()
                if roi_area_ratio < 0.5:  # ROI가 전체의 50% 미만이면 효율적
                    self.stats['roi_performance'] = 'excellent'
                elif roi_area_ratio < 0.8:
                    self.stats['roi_performance'] = 'good'
                else:
                    self.stats['roi_performance'] = 'poor'
                    
        except Exception as e:
            print(f"[ERROR] ROI 업데이트 실패: {e}")
    
    def _calculate_fallback_muzzle_roi(self, width: int, height: int) -> Tuple[int, int, int, int]:
        """Body Tracking 실패 시 대체 총구 ROI"""
        # 상체 중심으로 더 넓은 영역 설정
        center_x, center_y = width // 2, height // 3
        roi_size = 600  # 더 큰 영역
        
        x1 = max(0, center_x - roi_size // 2)
        y1 = max(0, center_y - roi_size // 2)
        x2 = min(width, center_x + roi_size // 2)
        y2 = min(height, center_y + roi_size // 2)
        
        return (x1, y1, x2, y2)
    
    def _calculate_roi_efficiency(self) -> float:
        """현재 ROI 효율성 계산 (전체 대비 ROI 면적 비율)"""
        try:
            total_pixels = 1920 * 1080  # HD 해상도
            
            # 스크린 ROI 면적
            screen_roi = getattr(self.laser_core, '_roi_regions', {}).get('screen')
            screen_area = 0
            if screen_roi:
                x1, y1, x2, y2 = screen_roi
                screen_area = (x2 - x1) * (y2 - y1)
            
            # 총구 ROI 면적  
            gun_roi = getattr(self.laser_core, '_roi_regions', {}).get('gun')
            gun_area = 0
            if gun_roi:
                x1, y1, x2, y2 = gun_roi
                gun_area = (x2 - x1) * (y2 - y1)
            
            # 전체 ROI 면적 비율
            total_roi_area = screen_area + gun_area
            efficiency = total_roi_area / total_pixels if total_pixels > 0 else 1.0
            
            return efficiency
            
        except Exception:
            return 1.0  # 계산 실패 시 최악값 반환
    
    def _apply_performance_optimizations(self):
        """성능 최적화 설정 적용"""
        try:
            print("[PERF] ⚡ 성능 최적화 설정 적용 중...")
            
            # 1. Azure Kinect 설정 최적화
            if hasattr(self, 'azure_kinect_config'):
                # 깊이 필터링 활성화 (불필요한 픽셀 제거)
                self.azure_kinect_config['use_depth_filtering'] = True
                self.azure_kinect_config['depth_filter_range'] = (500, 8000)  # 0.5m ~ 8m
                
                # 깊이 해상도 최적화 (NFOV 2x2 BINNED for dual stability)
                self.azure_kinect_config['depth_mode'] = 'NFOV_2X2BINNED'
                
                # 프레임 스킵 설정 (안정성 향상)
                self.azure_kinect_config['frame_skip_threshold'] = 2
            
            # 2. 레이저 검출 최적화
            if hasattr(self, 'laser_core'):
                # OpenCV 최적화 플래그 설정
                import cv2
                cv2.setUseOptimized(True)
                cv2.setNumThreads(4)  # 멀티스레드 활용
                
                # 불필요한 검출 단계 비활성화
                if hasattr(self.laser_core, 'config'):
                    self.laser_core.config['enable_morphology'] = False  # 형태학적 연산 비활성화
                    self.laser_core.config['enable_contour_filter'] = True  # 필수 필터만 유지
            
            # 3. 프레임 처리 최적화
            self.roi_update_interval = 10  # ROI 업데이트 주기 연장 (5→10프레임)
            
            # 4. 통계 수집 최적화
            if hasattr(self, 'stats'):
                self.stats['performance_optimizations_applied'] = True
                self.stats['target_fps'] = 50
                self.stats['roi_efficiency_target'] = 0.5  # 50% 이하 ROI 면적 목표
            
            # 해상도 정보 표시를 실제 캡처 해상도 기준으로 출력
            try:
                cam_info = self.camera_manager.get_camera_info()
                cap_w, cap_h = cam_info.get('resolution', (1920, 1080))
            except Exception:
                cap_w, cap_h = (1920, 1080)

            print("[PERF] ✅ 성능 최적화 완료:")
            print(f"   • 캡처 해상도 ({cap_w}x{cap_h})")
            print("   • 지능형 ROI 활성화")  
            print("   • 깊이 필터링 활성화")
            print("   • OpenCV 멀티스레드")
            print("   • 불필요한 연산 제거")
            print("   • 목표: 50+ FPS")
            
        except Exception as e:
            print(f"[ERROR] 성능 최적화 실패: {e}")
    
    def get_performance_stats(self) -> dict:
        """성능 통계 반환"""
        try:
            stats = {}
            
            # 기본 FPS 계산
            if hasattr(self, 'stats') and 'start_time' in self.stats:
                runtime = time.time() - self.stats['start_time']
                if runtime > 0 and 'frames_processed' in self.stats:
                    current_fps = self.stats['frames_processed'] / runtime
                    stats['current_fps'] = round(current_fps, 1)
                    stats['target_fps'] = 50
                    stats['fps_achievement'] = f"{(current_fps/50)*100:.1f}%" if current_fps > 0 else "0%"
            
            # ROI 효율성
            roi_efficiency = self._calculate_roi_efficiency()
            stats['roi_efficiency'] = f"{roi_efficiency*100:.1f}%"
            stats['roi_performance'] = self.stats.get('roi_performance', 'unknown') if hasattr(self, 'stats') else 'unknown'
            
            # 메모리 사용량 (대략적)
            pixel_reduction = (1 - roi_efficiency) * 100
            stats['pixel_reduction'] = f"{pixel_reduction:.1f}%"
            stats['memory_saved'] = f"{pixel_reduction * 0.8:.1f}%"  # 대략적 메모리 절약
            
            return stats
            
        except Exception as e:
            return {'error': str(e)}

# 기존 시스템과의 호환성을 위한 래퍼 클래스
class MainController(MainControllerAzureKinect):
    """기존 MainController와의 호환성을 위한 래퍼"""
    
    def __init__(self, camera_id: int = 0):
        # 기존 방식: camera_id로 웹캠 사용
        super().__init__(camera_type="webcam", device_id=camera_id)