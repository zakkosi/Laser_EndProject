#!/usr/bin/env python3
"""
Body Tracking 보조 워커 (Azure Kinect BT SDK 통합)

목적:
- Azure Kinect Body Tracking SDK를 별도 스레드에서 실행하여
  손목-검지 벡터 기반 총구 검출의 ROI/연속성/스코어 융합에 사용할 정보를 제공한다.

주요 기능:
- Azure Kinect BT SDK 실제 연동 
- 손목(WRIST) → 검지끝(INDEX_TIP) 3D 좌표 추적
- 총구 위치/방향 벡터 계산
- 스레드 안전 결과 제공
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import threading
import time
import logging
import numpy as np

# Azure Kinect Body Tracking SDK
try:
    import pykinect_azure as pykinect
    from pykinect_azure import k4a
    from pykinect_azure import k4abt
    AZURE_KINECT_BT_AVAILABLE = True
    print("[INFO] Azure Kinect Body Tracking SDK 로드 성공")
except ImportError as e:
    AZURE_KINECT_BT_AVAILABLE = False
    print(f"[WARN] Azure Kinect Body Tracking SDK 로드 실패: {e}")
    print("[WARN] Body Tracking 기능이 비활성화됩니다")


@dataclass
class BTResult:
    """Body Tracking 결과 (손목-검지 벡터 기반 총구 추정)

    Attributes:
        person_id: 추적된 인원 식별자
        wrist_2d: 손목 2D 좌표(px)
        index_tip_2d: 검지 끝 2D 좌표(px)
        wrist_3d: 손목 3D 좌표(mm)
        index_tip_3d: 검지 끝 3D 좌표(mm)
        
        # 총구 추정 결과
        estimated_muzzle_3d: 손목-검지 벡터 기반 총구 3D 위치(mm)
        muzzle_direction: 총구 방향 벡터(정규화됨)
        gun_length_mm: 사용된 총기 길이(mm)
        
        confidence: 전체 신뢰도(0~1)
        joint_confidence: 각 관절별 신뢰도 [손목, 검지]
        timestamp: Unix 타임스탬프(초)
    """
    person_id: int
    wrist_2d: Optional[Tuple[int, int]]
    index_tip_2d: Optional[Tuple[int, int]]
    wrist_3d: Optional[Tuple[float, float, float]]
    index_tip_3d: Optional[Tuple[float, float, float]]
    
    # 총구 추정
    estimated_muzzle_3d: Optional[Tuple[float, float, float]]
    muzzle_direction: Optional[Tuple[float, float, float]]
    gun_length_mm: float
    
    confidence: float
    joint_confidence: Optional[List[float]]
    timestamp: float


class BodyTrackingWorker:
    """Azure Kinect Body Tracking 워커 (실제 SDK 연동)
    
    손목-검지 벡터 기반 총구 위치/방향 추정을 위한 Body Tracking 시스템
    """

    def __init__(self, 
                 camera_id: int = 0,
                 target_fps: int = 12,
                 gun_length_mm: float = 200.0,
                 confidence_threshold: float = 0.6,
                 camera_intrinsics: Optional[Dict] = None):
        """
        Args:
            camera_id: Azure Kinect 카메라 ID (총구 추적용 카메라)
            target_fps: Body Tracking 목표 FPS
            gun_length_mm: 총기 길이 (손목-검지 벡터 연장용)
            confidence_threshold: 최소 신뢰도 임계값
        """
        # 실행 파라미터
        self.camera_id = camera_id
        self.target_fps = max(1, min(30, int(target_fps)))
        self.gun_length_mm = gun_length_mm
        self.confidence_threshold = confidence_threshold
        
        # 카메라 내부 파라미터 (3D→2D 프로젝션용)
        self.camera_intrinsics = camera_intrinsics or {
            'fx': 920.0, 'fy': 920.0,  # Azure Kinect HD 기본값
            'cx': 960.0, 'cy': 540.0   # 1920x1080 해상도 중심점
        }

        # Azure Kinect 객체들
        self.device: Optional[k4a.Device] = None
        self.tracker: Optional[k4abt.Tracker] = None
        # 외부 캡처(공유 캡처) 모드: CameraManager에서 받은 raw_capture를 사용
        self.external_capture_mode: bool = False
        self.external_capture_queue: List[object] = []

        # 스레드/동기화
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # 최신 결과 및 통계
        self._latest_result: Optional[BTResult] = None
        self._frame_count = 0
        self._detection_count = 0
        
        # 로깅
        self.logger = logging.getLogger(__name__)
        
        # SDK 가용성 체크
        if not AZURE_KINECT_BT_AVAILABLE:
            self.logger.warning("Azure Kinect BT SDK가 없어 Body Tracking이 비활성화됩니다")
            
    def _project_3d_to_2d(self, point_3d: Tuple[float, float, float]) -> Optional[Tuple[int, int]]:
        """3D 카메라 좌표를 2D 픽셀 좌표로 변환
        
        Azure Kinect Body Tracking 3D 좌표 → 2D 픽셀 좌표 변환
        
        Args:
            point_3d: (x, y, z) 3D 좌표 (mm 단위)
            
        Returns:
            (u, v) 2D 픽셀 좌표 또는 None (유효하지 않은 경우)
        """
        try:
            x, y, z = point_3d
            
            # 유효한 깊이 체크
            if z <= 0 or z > 8000:  # 8m 이상은 무시
                return None
                
            # 핀홀 카메라 모델 사용
            fx = self.camera_intrinsics['fx']
            fy = self.camera_intrinsics['fy'] 
            cx = self.camera_intrinsics['cx']
            cy = self.camera_intrinsics['cy']
            
            # 프로젝션 계산
            u = int(fx * x / z + cx)
            v = int(fy * y / z + cy)
            
            # 화면 경계 체크 (1920x1080 가정)
            if 0 <= u < 1920 and 0 <= v < 1080:
                return (u, v)
            else:
                return None
                
        except (ZeroDivisionError, ValueError, TypeError) as e:
            self.logger.error(f"3D→2D 프로젝션 실패: {e}")
            return None

    def _calculate_muzzle_from_hand(self, 
                                  wrist_3d: Tuple[float, float, float],
                                  index_tip_3d: Tuple[float, float, float]) -> Tuple[Optional[Tuple[float, float, float]], Optional[Tuple[float, float, float]]]:
        """손목-검지 벡터로 총구 위치와 방향 계산
        
        Args:
            wrist_3d: 손목 3D 좌표 (mm)
            index_tip_3d: 검지 끝 3D 좌표 (mm)
            
        Returns:
            (estimated_muzzle_3d, muzzle_direction): 총구 위치와 방향 벡터
        """
        try:
            # 벡터 계산
            wrist_np = np.array(wrist_3d)
            index_np = np.array(index_tip_3d)
            
            # 손목 → 검지 방향 벡터
            direction = index_np - wrist_np
            distance = np.linalg.norm(direction)
            
            if distance < 10.0:  # 10mm 미만이면 너무 가까움
                return None, None
                
            # 정규화된 방향 벡터
            direction_normalized = direction / distance
            
            # 총구 위치 = 검지 + 방향벡터 * 총기길이
            estimated_muzzle = index_np + direction_normalized * self.gun_length_mm
            
            return tuple(estimated_muzzle.tolist()), tuple(direction_normalized.tolist())
            
        except Exception as e:
            self.logger.error(f"총구 위치 계산 실패: {e}")
            return None, None

    def start(self):
        """워커 스레드 시작"""
        if not AZURE_KINECT_BT_AVAILABLE:
            self.logger.error("Azure Kinect BT SDK가 없어 시작할 수 없습니다")
            return False
            
        if self._thread and self._thread.is_alive():
            return True
            
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self.logger.info(f"Body Tracking 워커 시작 (카메라 ID: {self.camera_id}, FPS: {self.target_fps})")
        return True

    def stop(self):
        """워커 스레드 중지"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            
        # Azure Kinect 리소스 정리
        self._cleanup_kinect()
        self.logger.info("Body Tracking 워커 중지됨")

    def get_latest_result(self) -> Optional[BTResult]:
        """최신 BT 결과 반환(스레드 세이프)"""
        with self._lock:
            return self._latest_result
            
    def get_detection_stats(self) -> dict:
        """Body Tracking 통계 반환"""
        with self._lock:
            detection_rate = self._detection_count / max(1, self._frame_count) * 100
            return {
                'total_frames': self._frame_count,
                'detections': self._detection_count,
                'detection_rate_percent': detection_rate
            }

    # 내부 구현 -------------------------------------------------------------
    def _init_kinect(self) -> bool:
        """Azure Kinect 초기화"""
        try:
            # 외부 캡처 모드에서는 디바이스를 직접 열지 않음
            if not self.external_capture_mode:
                self.logger.info(f"Azure Kinect 초기화 시작 (카메라 ID: {self.camera_id})")
                self.device = k4a.Device.open(self.camera_id)
                if not self.device:
                    self.logger.error(f"Azure Kinect 장치 {self.camera_id} 열기 실패")
                    return False
                device_config = k4a.DeviceConfiguration(
                    color_format=k4a.ImageFormat.COLOR_BGRA32,
                    color_resolution=k4a.ColorResolution.RES_720P,
                    depth_mode=k4a.DepthMode.NFOV_UNBINNED,
                    camera_fps=k4a.FramesPerSecond.FPS_30,
                    synchronized_images_only=True
                )
                self.device.start_cameras(device_config)
            
            # Body Tracker 초기화 (공유/내부 모두 동일)
            tracker_config = k4abt.TrackerConfiguration()
            tracker_config.processing_mode = k4abt.TrackerProcessingMode.GPU  # GPU 가속 사용
            tracker_config.model_type = k4abt.ModelType.LITE  # 경량 모델 사용
            
            self.tracker = k4abt.Tracker(tracker_config)
            if not self.tracker:
                self.logger.error("Body Tracker 초기화 실패")
                return False
                
            self.logger.info("Azure Kinect Body Tracking 초기화 성공")
            return True
            
        except Exception as e:
            self.logger.error(f"Azure Kinect 초기화 실패: {e}")
            return False
    
    def _cleanup_kinect(self):
        """Azure Kinect 리소스 정리"""
        try:
            if self.tracker:
                self.tracker.shutdown()
                self.tracker = None
                
            if self.device:
                self.device.stop_cameras()
                self.device.close()
                self.device = None
                
        except Exception as e:
            self.logger.error(f"Azure Kinect 정리 중 오류: {e}")

    def _run_loop(self):
        """실제 Body Tracking 루프"""
        # Azure Kinect 초기화
        if not self._init_kinect():
            self.logger.error("Body Tracking 초기화 실패 - 루프 종료")
            return
            
        period = 1.0 / float(self.target_fps)
        self.logger.info("Body Tracking 메인 루프 시작")
        
        try:
            while not self._stop_event.is_set():
                loop_start = time.time()
                
                # 카메라에서 프레임 캡처
                if self.external_capture_mode:
                    if not self.external_capture_queue:
                        time.sleep(0.001)
                        continue
                    capture = self.external_capture_queue.pop(0)
                else:
                    capture = self.device.get_capture()
                if capture.depth_image is None:
                    continue
                    
                self._frame_count += 1
                
                # Body Tracking 수행
                body_frame = self.tracker.enqueue_capture(capture)
                if body_frame is None:
                    continue
                    
                # Body 검출 결과 처리
                bt_result = self._process_bodies(body_frame)
                
                # 결과 업데이트 (스레드 안전)
                with self._lock:
                    self._latest_result = bt_result
                    if bt_result is not None:
                        self._detection_count += 1
                
                # 프레임률 제어
                elapsed = time.time() - loop_start
                if elapsed < period:
                    time.sleep(period - elapsed)
                    
        except Exception as e:
            self.logger.error(f"Body Tracking 루프 오류: {e}")
        finally:
            if not self.external_capture_mode:
                self._cleanup_kinect()
            
    def _process_bodies(self, body_frame) -> Optional[BTResult]:
        """Body Tracking 결과 처리"""
        try:
            num_bodies = body_frame.get_num_bodies()
            if num_bodies == 0:
                return None
                
            # 첫 번째 검출된 사람 사용 (다중 사용자는 추후 확장)
            body = body_frame.get_body(0)
            
            # 필요한 관절점들 (손목과 검지 끝)
            joints = body.joints
            
            # 오른손 우선, 없으면 왼손 사용
            wrist_joint = None
            index_joint = None
            
            # 오른손 체크
            if (k4abt.JointType.WRIST_RIGHT in joints and 
                k4abt.JointType.HANDTIP_RIGHT in joints):
                wrist_joint = joints[k4abt.JointType.WRIST_RIGHT]
                index_joint = joints[k4abt.JointType.HANDTIP_RIGHT]
            # 왼손 체크
            elif (k4abt.JointType.WRIST_LEFT in joints and 
                  k4abt.JointType.HANDTIP_LEFT in joints):
                wrist_joint = joints[k4abt.JointType.WRIST_LEFT]
                index_joint = joints[k4abt.JointType.HANDTIP_LEFT]
            else:
                return None  # 필요한 관절점들이 검출되지 않음
                
            # 신뢰도 체크
            wrist_conf = wrist_joint.confidence_level.value / 4.0  # 정규화 (0~1)
            index_conf = index_joint.confidence_level.value / 4.0
            avg_confidence = (wrist_conf + index_conf) / 2.0
            
            if avg_confidence < self.confidence_threshold:
                return None  # 신뢰도가 너무 낮음
                
            # 3D 좌표 추출 (Azure Kinect는 mm 단위)
            wrist_3d = (wrist_joint.position.x, wrist_joint.position.y, wrist_joint.position.z)
            index_3d = (index_joint.position.x, index_joint.position.y, index_joint.position.z)
            
            # 총구 위치와 방향 계산
            estimated_muzzle, muzzle_direction = self._calculate_muzzle_from_hand(wrist_3d, index_3d)
            
            if estimated_muzzle is None:
                return None  # 총구 계산 실패
                
                # 3D → 2D 프로젝션 계산 (Azure Kinect calibration 사용)
            wrist_2d = self._project_3d_to_2d(wrist_3d)
            index_2d = self._project_3d_to_2d(index_3d)
            
            # BTResult 생성
            return BTResult(
                person_id=body.id,
                wrist_2d=wrist_2d,
                index_tip_2d=index_2d,
                wrist_3d=wrist_3d,
                index_tip_3d=index_3d,
                estimated_muzzle_3d=estimated_muzzle,
                muzzle_direction=muzzle_direction,
                gun_length_mm=self.gun_length_mm,
                confidence=avg_confidence,
                joint_confidence=[wrist_conf, index_conf],
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Body 처리 중 오류: {e}")
            return None

