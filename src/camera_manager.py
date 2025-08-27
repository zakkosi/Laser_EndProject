#!/usr/bin/env python3
"""
Virtra 카메라 관리 모듈 (Camera Manager Module)
기존 모듈화 구조에서 다양한 카메라 타입 지원

지원 카메라:
1. 웹캠 (기존)
2. ZED (기존) 
3. Azure Kinect DK (신규)
4. 듀얼 Azure Kinect DK (신규)
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

# Azure Kinect 관련 import
try:
    import pykinect_azure as pykinect
    KINECT_AVAILABLE = True
except ImportError:
    KINECT_AVAILABLE = False
    pykinect = None

# ZED 관련 import
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    sl = None

class CameraType(Enum):
    """카메라 타입 열거형"""
    WEBCAM = "webcam"
    ZED = "zed"
    AZURE_KINECT = "azure_kinect"
    DUAL_AZURE_KINECT = "dual_azure_kinect"

@dataclass
class CameraConfig:
    """카메라 설정 정보"""
    camera_type: CameraType
    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    use_depth: bool = False
    use_4k: bool = False
    # Azure Kinect 전용 설정
    azure_kinect_config: Optional[Dict] = None

@dataclass
class CameraFrame:
    """카메라 프레임 데이터"""
    color_frame: np.ndarray
    depth_frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame_id: int = 0
    camera_info: Optional[Dict] = None
    # Azure Kinect 원본 캡처(외부 모듈 공유용). 다른 카메라 타입에서는 None 유지
    raw_capture: Optional[object] = None

class CameraManager:
    """
    통합 카메라 관리자
    기존 모듈화 구조와 호환되는 카메라 인터페이스 제공
    """
    
    def __init__(self, config: CameraConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 카메라 객체들
        self.cap = None  # 웹캠용
        self.zed = None  # ZED용
        self.kinect_primary = None  # Azure Kinect 기본
        self.kinect_secondary = None  # Azure Kinect 보조 (듀얼용)
        
        # ZED 전용 객체들
        self.zed_image = None
        self.zed_depth = None
        self.zed_runtime_params = None
        
        # 상태 정보
        self.is_initialized = False
        self.frame_count = 0
        self.last_frame_time = 0.0
        
        print(f"[INFO] Camera Manager Init: {config.camera_type.value}")
    
    def initialize(self) -> bool:
        """카메라 초기화"""
        try:
            if self.config.camera_type == CameraType.WEBCAM:
                return self._initialize_webcam()
            elif self.config.camera_type == CameraType.ZED:
                return self._initialize_zed()
            elif self.config.camera_type == CameraType.AZURE_KINECT:
                return self._initialize_azure_kinect()
            elif self.config.camera_type == CameraType.DUAL_AZURE_KINECT:
                return self._initialize_dual_azure_kinect()
            else:
                self.logger.error(f"지원하지 않는 카메라 타입: {self.config.camera_type}")
                return False
        except Exception as e:
            self.logger.error(f"카메라 초기화 실패: {e}")
            return False
    
    def _initialize_webcam(self) -> bool:
        """웹캠 초기화 (기존 main_controller 로직)"""
        try:
            print(f"[INFO] Initializing Webcam {self.config.device_id}...")
            
            self.cap = cv2.VideoCapture(self.config.device_id, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.config.device_id)
                if not self.cap.isOpened():
                    return False
            
            # 해상도 설정
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 안정화
            for _ in range(10):
                ret, frame = self.cap.read()
                time.sleep(0.1)
            
            self.is_initialized = True
            print("[INFO] Webcam Initialized Successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"웹캠 초기화 실패: {e}")
            return False
    
    def _initialize_zed(self) -> bool:
        """ZED 초기화 (기존 로직)"""
        if not ZED_AVAILABLE:
            self.logger.error("ZED SDK를 사용할 수 없습니다")
            return False
        
        try:
            print("[INFO] Initializing ZED Camera...")
            
            self.zed = sl.Camera()
            
            init_params = sl.InitParameters()
            init_params.camera_resolution = sl.RESOLUTION.HD720
            init_params.camera_fps = 30
            init_params.depth_mode = sl.DEPTH_MODE.NEURAL
            
            err = self.zed.open(init_params)
            if err != sl.ERROR_CODE.SUCCESS:
                return False
            
            self.zed_image = sl.Mat()
            self.zed_depth = sl.Mat()
            self.zed_runtime_params = sl.RuntimeParameters()
            
            # 안정화
            for _ in range(10):
                if self.zed.grab(self.zed_runtime_params) == sl.ERROR_CODE.SUCCESS:
                    self.zed.retrieve_image(self.zed_image, sl.VIEW.LEFT)
                    if self.config.use_depth:
                        self.zed.retrieve_measure(self.zed_depth, sl.MEASURE.DEPTH)
                time.sleep(0.1)
            
            self.is_initialized = True
            print("[INFO] ZED Camera Initialized Successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"ZED 초기화 실패: {e}")
            return False
    
    def _initialize_azure_kinect(self) -> bool:
        """Azure Kinect 단일 초기화"""
        if not KINECT_AVAILABLE:
            self.logger.error("Azure Kinect SDK를 사용할 수 없습니다")
            print("[INFO] Install with: pip install pykinect-azure")
            return False
        
        try:
            print(f"[INFO] Initializing Azure Kinect {self.config.device_id}...")
            
            pykinect.initialize_libraries()
            
            device_config = pykinect.default_configuration
            
            # 해상도 설정
            if self.config.use_4k:
                device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_3072P
                device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_15
                print("[INFO] 4K Mode (3072p) Enabled")
            else:
                device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
                device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
                print("[INFO] HD Mode (1080p) Enabled")
            
            device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
            device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED
            device_config.synchronized_images_only = True
            
            self.kinect_primary = pykinect.start_device(config=device_config, device_index=self.config.device_id)
            
            # 안정화
            for _ in range(15):
                capture = self.kinect_primary.update()
                if capture.get_color_image() is not None:
                    break
                time.sleep(0.1)
            
            self.is_initialized = True
            print("[INFO] Azure Kinect Initialized Successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Azure Kinect 초기화 실패: {e}")
            return False
    
    def _initialize_dual_azure_kinect(self) -> bool:
        """듀얼 Azure Kinect 초기화"""
        if not KINECT_AVAILABLE:
            self.logger.error("Azure Kinect SDK를 사용할 수 없습니다")
            return False
        
        try:
            print("[INFO] Initializing Dual Azure Kinect...")
            
            pykinect.initialize_libraries()
            
            device_config = pykinect.default_configuration
            
            if self.config.use_4k:
                # 듀얼 환경에서 4K는 대역폭 과부하 유발 → 1080P로 강제 다운스케일
                device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
                device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
            else:
                # 듀얼 기본은 720P@30으로 대역폭 여유 확보
                device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
                device_config.camera_fps = pykinect.K4A_FRAMES_PER_SECOND_30
            
            device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
            # NFOV_BINNED은 계산부하와 데이터량을 추가로 낮춤(듀얼에서 권장)
            device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED
            # 강한 동기화는 내부 큐 적체를 유발 → 비동기 수집 후 사용자 레벨에서 시간정렬
            device_config.synchronized_images_only = False
            device_config.wired_sync_mode = pykinect.K4A_WIRED_SYNC_MODE_STANDALONE  # 독립 모드
            
            # 기본 디바이스 (ID: 0)
            try:
                self.kinect_primary = pykinect.start_device(config=device_config, device_index=0)
                print("[INFO] Primary Azure Kinect (ID: 0) Initialized")
            except Exception as e:
                print(f"[ERROR] Primary Azure Kinect Init Failed: {e}")
                return False
            
            # 보조 디바이스 (ID: 1)
            try:
                self.kinect_secondary = pykinect.start_device(config=device_config, device_index=1)
                print("[INFO] Secondary Azure Kinect (ID: 1) Initialized")
            except Exception as e:
                print(f"[WARN] Secondary Azure Kinect Init Failed: {e}")
                print("[INFO] Running in Single Kinect Mode")
                self.kinect_secondary = None
            
            # 안정화
            for _ in range(15):
                if self.kinect_primary:
                    _ = self.kinect_primary.update()
                if self.kinect_secondary:
                    _ = self.kinect_secondary.update()
                time.sleep(0.05)
            
            device_count = 1 + (1 if self.kinect_secondary else 0)
            self.is_initialized = True
            print(f"[INFO] {device_count} Azure Kinect Device(s) Initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"듀얼 Azure Kinect 초기화 실패: {e}")
            return False
    
    def capture_frame(self) -> Optional[CameraFrame]:
        """프레임 캡처 (통합 인터페이스)"""
        if not self.is_initialized:
            return None
        
        try:
            self.frame_count += 1
            current_time = time.time()
            
            if self.config.camera_type == CameraType.WEBCAM:
                return self._capture_webcam_frame(current_time)
            elif self.config.camera_type == CameraType.ZED:
                return self._capture_zed_frame(current_time)
            elif self.config.camera_type == CameraType.AZURE_KINECT:
                return self._capture_azure_kinect_frame(current_time)
            elif self.config.camera_type == CameraType.DUAL_AZURE_KINECT:
                return self._capture_dual_azure_kinect_frame(current_time)
            
            return None
            
        except Exception as e:
            self.logger.error(f"프레임 캡처 실패: {e}")
            return None
    
    def _capture_webcam_frame(self, timestamp: float) -> Optional[CameraFrame]:
        """웹캠 프레임 캡처"""
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        
        return CameraFrame(
            color_frame=frame,
            depth_frame=None,
            timestamp=timestamp,
            frame_id=self.frame_count,
            camera_info={'type': 'webcam', 'device_id': self.config.device_id}
        )
    
    def _capture_zed_frame(self, timestamp: float) -> Optional[CameraFrame]:
        """ZED 프레임 캡처"""
        if self.zed.grab(self.zed_runtime_params) != sl.ERROR_CODE.SUCCESS:
            return None
        
        self.zed.retrieve_image(self.zed_image, sl.VIEW.LEFT)
        color_frame = self.zed_image.get_data()
        
        # RGBA to BGR 변환
        if len(color_frame.shape) == 3 and color_frame.shape[2] == 4:
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_RGBA2BGR)
        
        depth_frame = None
        if self.config.use_depth:
            self.zed.retrieve_measure(self.zed_depth, sl.MEASURE.DEPTH)
            depth_frame = self.zed_depth.get_data()
        
        return CameraFrame(
            color_frame=color_frame,
            depth_frame=depth_frame,
            timestamp=timestamp,
            frame_id=self.frame_count,
            camera_info={'type': 'zed', 'has_depth': depth_frame is not None}
        )
    
    def _capture_azure_kinect_frame(self, timestamp: float) -> Optional[CameraFrame]:
        """Azure Kinect 프레임 캡처"""
        capture = self.kinect_primary.update()
        
        # Azure Kinect API는 (success, image) 튜플을 반환
        success, color_frame = capture.get_color_image()
        if not success or color_frame is None:
            return None
        
        # 색상 프레임 처리
        if len(color_frame.shape) == 3 and color_frame.shape[2] == 4:
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
        
        # 깊이 프레임 처리
        depth_frame = None
        if self.config.use_depth:
            depth_success, depth_frame = capture.get_depth_image()
            if not depth_success or depth_frame is None:
                depth_frame = None
        
        return CameraFrame(
            color_frame=color_frame,
            depth_frame=depth_frame,
            timestamp=timestamp,
            frame_id=self.frame_count,
            camera_info={
                'type': 'azure_kinect',
                'device_id': self.config.device_id,
                'has_depth': depth_frame is not None,
                'resolution_4k': self.config.use_4k
            },
            raw_capture=capture
        )
    
    def _capture_dual_azure_kinect_frame(self, timestamp: float) -> Optional[CameraFrame]:
        """듀얼 Azure Kinect 프레임 캡처 (기본 카메라만 반환, 보조는 별도 처리)"""
        # 최신 프레임만 소비: 내부 큐에 쌓인 오래된 프레임을 드레인하여 capturesync_drop 방지
        capture_primary = None
        for _ in range(3):
            cap = self.kinect_primary.update()
            if cap is not None:
                capture_primary = cap
        if capture_primary is None:
            return None
        
        # Azure Kinect API는 (success, image) 튜플을 반환
        success, color_frame = capture_primary.get_color_image()
        if not success or color_frame is None:
            return None
        
        # 색상 프레임 처리
        if len(color_frame.shape) == 3 and color_frame.shape[2] == 4:
            color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
        
        # 깊이 프레임 처리
        depth_frame = None
        if self.config.use_depth:
            # 깊이도 최신 것만 취득(컬러와 완전 동기 강제하지 않음)
            depth_success, depth_frame = capture_primary.get_depth_image()
            if not depth_success or depth_frame is None:
                depth_frame = None
        
        # 보조 카메라 정보도 포함
        secondary_available = self.kinect_secondary is not None
        
        return CameraFrame(
            color_frame=color_frame,
            depth_frame=depth_frame,
            timestamp=timestamp,
            frame_id=self.frame_count,
            camera_info={
                'type': 'dual_azure_kinect',
                'primary_device_id': 0,
                'secondary_available': secondary_available,
                'has_depth': depth_frame is not None,
                'resolution_4k': self.config.use_4k
            },
            raw_capture=capture_primary
        )
    
    def capture_secondary_frame(self) -> Optional[CameraFrame]:
        """보조 Azure Kinect 프레임 캡처 (듀얼 모드 전용)"""
        if (self.config.camera_type != CameraType.DUAL_AZURE_KINECT or 
            self.kinect_secondary is None):
            return None
        
        try:
            capture_secondary = self.kinect_secondary.update()
            
            # Azure Kinect API는 (success, image) 튜플을 반환
            success, color_frame = capture_secondary.get_color_image()
            if not success or color_frame is None:
                return None
            
            # 색상 프레임 처리
            if len(color_frame.shape) == 3 and color_frame.shape[2] == 4:
                color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGRA2BGR)
            
            # 깊이 프레임 처리
            depth_frame = None
            if self.config.use_depth:
                depth_success, depth_frame = capture_secondary.get_depth_image()
                if not depth_success:
                    depth_frame = None
            
            return CameraFrame(
                color_frame=color_frame,
                depth_frame=depth_frame,
                timestamp=time.time(),
                frame_id=self.frame_count,
                camera_info={
                    'type': 'dual_azure_kinect_secondary',
                    'device_id': 1,
                    'has_depth': depth_frame is not None,
                    'resolution_4k': self.config.use_4k
                },
                raw_capture=capture_secondary
            )
            
        except Exception as e:
            self.logger.error(f"보조 Azure Kinect 프레임 캡처 실패: {e}")
            return None
    
    def get_camera_info(self) -> Dict[str, Any]:
        """카메라 정보 반환"""
        info = {
            'camera_type': self.config.camera_type.value,
            'device_id': self.config.device_id,
            'resolution': (self.config.width, self.config.height),
            'fps': self.config.fps,
            'use_depth': self.config.use_depth,
            'is_initialized': self.is_initialized,
            'frame_count': self.frame_count
        }
        
        if self.config.camera_type == CameraType.AZURE_KINECT:
            info.update({
                'use_4k': self.config.use_4k,
                'kinect_available': KINECT_AVAILABLE
            })
        elif self.config.camera_type == CameraType.DUAL_AZURE_KINECT:
            info.update({
                'use_4k': self.config.use_4k,
                'kinect_available': KINECT_AVAILABLE,
                'secondary_available': self.kinect_secondary is not None
            })
        elif self.config.camera_type == CameraType.ZED:
            info.update({
                'zed_available': ZED_AVAILABLE
            })
        
        return info
    
    def cleanup(self):
        """리소스 정리"""
        try:
            if self.cap:
                self.cap.release()
            if self.zed:
                self.zed.close()
            if self.kinect_primary:
                self.kinect_primary.close()
            if self.kinect_secondary:
                self.kinect_secondary.close()
            
            print("🧹 카메라 리소스 정리 완료")
            
        except Exception as e:
            self.logger.error(f"카메라 리소스 정리 실패: {e}")

def create_camera_manager(camera_type_str: str, **kwargs) -> CameraManager:
    """카메라 관리자 팩토리 함수"""
    try:
        camera_type = CameraType(camera_type_str.lower())
    except ValueError:
        raise ValueError(f"지원하지 않는 카메라 타입: {camera_type_str}")
    
    # 기본 설정
    config = CameraConfig(
        camera_type=camera_type,
        device_id=kwargs.get('device_id', 0),
        width=kwargs.get('width', 1280),
        height=kwargs.get('height', 720),
        fps=kwargs.get('fps', 30),
        use_depth=kwargs.get('use_depth', True),
        use_4k=kwargs.get('use_4k', False)
    )
    
    # Azure Kinect 해상도/프레임 기본 설정(듀얼은 720p@30 권장)
    if camera_type in [CameraType.AZURE_KINECT, CameraType.DUAL_AZURE_KINECT]:
        if camera_type == CameraType.DUAL_AZURE_KINECT:
            config.width = 1280
            config.height = 720
            config.fps = 30
        else:
            if kwargs.get('use_4k', False):
                config.width = 3840
                config.height = 2160
                config.fps = 15
            else:
                config.width = 1920
                config.height = 1080
                config.fps = 30
    
    return CameraManager(config)

# 기존 시스템과의 호환성을 위한 간편 함수들
def create_webcam_manager(device_id: int = 0) -> CameraManager:
    """웹캠 관리자 생성"""
    return create_camera_manager("webcam", device_id=device_id, use_depth=False)

def create_zed_manager() -> CameraManager:
    """ZED 관리자 생성"""
    return create_camera_manager("zed", use_depth=True)

def create_azure_kinect_manager(device_id: int = 0, use_4k: bool = False) -> CameraManager:
    """Azure Kinect 관리자 생성"""
    return create_camera_manager("azure_kinect", device_id=device_id, use_depth=True, use_4k=use_4k)

def create_dual_azure_kinect_manager(use_4k: bool = False) -> CameraManager:
    """듀얼 Azure Kinect 관리자 생성"""
    return create_camera_manager("dual_azure_kinect", use_depth=True, use_4k=use_4k)