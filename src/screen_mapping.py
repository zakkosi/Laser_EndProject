"""
Virtra 스크린 매핑 모듈 (Screen Mapping Module)
원본 adaptive_laser_detector.py에서 2D 스크린 매핑 시스템과 Unity 통신 추출

핵심 기능:
- 2D 스크린 캘리브레이션 시스템
- Unity UDP 통신 (JSON 데이터 전송)
- 스크린 좌표 변환 (픽셀 → 스크린 → Unity)
- 캘리브레이션 데이터 관리
- 실시간 매핑 상태 관리
"""

import cv2
import numpy as np
import time
import logging
import json
import socket
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# 기존 스크린 캘리브레이션 시스템 임포트
from screen_calibration_system import ScreenCalibrationSystem


@dataclass
class MappingResult:
    """매핑 결과 데이터"""
    success: bool
    screen_coordinate: Optional[Tuple[float, float]]
    unity_coordinate: Optional[Tuple[float, float]]
    in_calibration_area: bool
    error_message: Optional[str] = None


class ScreenMapper:
    """
    2D 스크린 매핑 시스템
    
    담당 기능:
    - 2D 스크린 캘리브레이션 관리
    - Unity UDP 통신 및 데이터 전송
    - 좌표 변환 (픽셀 → 스크린 → Unity)
    - 캘리브레이션 상태 관리
    - 실시간 매핑 결과 처리
    """
    
    def __init__(self):
        """스크린 매퍼 초기화"""
        # 2D 스크린 캘리브레이션 시스템 (원본과 동일)
        self.screen_calibrator = ScreenCalibrationSystem()
        self.enable_2d_mapping = False  # 2D 매핑 활성화 플래그
        self.calibration_mode = False   # 캘리브레이션 모드 플래그
        
        # Unity UDP 통신 (원본과 동일)
        self.enable_unity_communication = True  # Unity 통신 활성화
        self.unity_ip = "127.0.0.1"            # Unity IP
        self.unity_port = 12345                 # Unity 포트
        self.udp_socket = None                  # UDP 소켓
        
        # Vector Fusion System 통신 추가
        self.enable_vector_fusion = True        # Vector Fusion 통신 활성화
        self.vector_fusion_ip = "127.0.0.1"    # Vector Fusion IP
        self.vector_fusion_port = 9999          # Vector Fusion 포트
        
        # 통신 통계
        self.communication_stats = {
            'total_sent': 0,
            'successful_sent': 0,
            'failed_sent': 0,
            'last_send_time': 0,
            'average_latency': 0.0
        }
        
        # 매핑 통계
        self.mapping_stats = {
            'total_attempts': 0,
            'successful_mappings': 0,
            'out_of_bounds': 0,
            'calibration_accuracy': 0.0
        }
        
        # 로깅
        self.logger = logging.getLogger(__name__)
        
        # Unity 통신 설정
        self._setup_unity_communication()
        
        print("[INFO] 스크린 매핑 모듈 초기화 완료")
    
    def _setup_unity_communication(self):
        """Unity UDP 통신 설정 (원본과 동일)"""
        if not self.enable_unity_communication:
            return
        
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"[INFO] Unity UDP 통신 설정 완료: {self.unity_ip}:{self.unity_port}")
        except Exception as e:
            print(f"[ERROR] Unity UDP 통신 설정 실패: {e}")
            self.enable_unity_communication = False
    
    def start_screen_calibration(self) -> bool:
        """
        2D 스크린 캘리브레이션 시작 (원본 _start_screen_calibration 기반)
        """
        try:
            print("\n" + "="*60)
            print("📐 2D 스크린 캘리브레이션 시작")
            print("="*60)
            
            # 스크린 크기 입력받기
            print("실제 스크린 크기를 입력하세요:")
            width_input = input("가로 크기 (cm): ").strip()
            height_input = input("세로 크기 (cm): ").strip()
            
            try:
                screen_width = float(width_input)
                screen_height = float(height_input)
                
                if screen_width <= 0 or screen_height <= 0:
                    print("❌ 스크린 크기는 0보다 커야 합니다.")
                    return False
                
                # 캘리브레이션 시작
                self.screen_calibrator.start_calibration(screen_width, screen_height)
                self.calibration_mode = True
                
                print("\n🎯 이제 스크린의 4개 모서리를 순서대로 클릭하세요:")
                print("   1️⃣ 좌측상단 모서리")
                print("   2️⃣ 우측상단 모서리") 
                print("   3️⃣ 우측하단 모서리")
                print("   4️⃣ 좌측하단 모서리")
                print("\n💡 정확한 캘리브레이션을 위해 스크린 경계선 끝부분을 정밀하게 클릭하세요!")
                
                return True
                
            except ValueError:
                print("❌ 올바른 숫자를 입력해주세요.")
                return False
                
        except Exception as e:
            print(f"❌ 캘리브레이션 시작 실패: {e}")
            return False
    
    def process_calibration_click(self, x: int, y: int) -> bool:
        """
        캘리브레이션 클릭 처리
        
        Args:
            x, y: 클릭 좌표
            
        Returns:
            캘리브레이션 완료 여부
        """
        if not self.calibration_mode:
            return False
        
        try:
            # 스크린 캘리브레이터에 클릭 전달
            is_complete = self.screen_calibrator.add_corner_point(x, y)
            
            if is_complete:
                # 캘리브레이션 완료
                self.calibration_mode = False
                accuracy = self.screen_calibrator.calibration_data.calibration_accuracy
                
                print(f"\n🎉 2D 스크린 캘리브레이션 완료!")
                print(f"📊 정확도: {accuracy*100:.1f}%")
                print(f"💡 [P] 키를 눌러 2D 매핑을 활성화하세요!")
                
                # 통계 업데이트
                self.mapping_stats['calibration_accuracy'] = accuracy
                
                return True
            else:
                # 다음 점 입력 대기
                points_added = len(self.screen_calibrator.calibration_data.screen_corners_pixel)
                print(f"📍 모서리 {points_added}/4 완료")
                return False
                
        except Exception as e:
            print(f"❌ 캘리브레이션 클릭 처리 실패: {e}")
            return False
    
    def map_to_screen(self, position: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        픽셀 좌표를 스크린 좌표로 변환 (호환성 메서드)
        
        Args:
            position: (x, y) 픽셀 좌표
            
        Returns:
            (screen_x, screen_y) 스크린 좌표 (0~1 범위) 또는 None
        """
        try:
            if not self.screen_calibrator.is_calibrated:
                return None
            
            x, y = position
            screen_coord = self.screen_calibrator.pixel_to_screen_coordinate(x, y)
            return screen_coord
        except Exception as e:
            self.logger.error(f"스크린 매핑 실패: {e}")
            return None
    
    def map_pixel_to_unity(self, pixel_x: int, pixel_y: int) -> MappingResult:
        """
        픽셀 좌표를 Unity 좌표로 변환
        
        Args:
            pixel_x, pixel_y: 픽셀 좌표
            
        Returns:
            매핑 결과
        """
        self.mapping_stats['total_attempts'] += 1
        
        result = MappingResult(
            success=False,
            screen_coordinate=None,
            unity_coordinate=None,
            in_calibration_area=False
        )
        
        try:
            # 캘리브레이션 확인
            if not self.screen_calibrator.is_calibrated:
                result.error_message = "캘리브레이션이 완료되지 않음"
                return result
            
            # 픽셀 → 스크린 좌표 변환
            screen_coord = self.screen_calibrator.pixel_to_screen_coordinate(pixel_x, pixel_y)
            
            if screen_coord is None:
                result.error_message = "픽셀이 캘리브레이션 영역 밖"
                self.mapping_stats['out_of_bounds'] += 1
                return result
            
            # 스크린 → Unity 좌표 변환
            unity_coord = self.screen_calibrator.get_unity_coordinate(
                screen_coord[0], screen_coord[1]
            )
            
            if unity_coord is None:
                result.error_message = "Unity 좌표 변환 실패"
                return result
            
            # 성공
            result.success = True
            result.screen_coordinate = screen_coord
            result.unity_coordinate = unity_coord
            result.in_calibration_area = True
            
            self.mapping_stats['successful_mappings'] += 1
            
            return result
            
        except Exception as e:
            result.error_message = f"매핑 오류: {e}"
            self.logger.error(f"매핑 오류: {e}")
            return result
    
    def send_to_unity(self, laser_result) -> bool:
        """
        Unity로 레이저 검출 결과 전송 (원본 _send_to_unity 기반)
        
        Args:
            laser_result: LaserDetectionResult 객체
            
        Returns:
            전송 성공 여부
        """
        if not self.enable_unity_communication or self.udp_socket is None:
            return False
        
        self.communication_stats['total_sent'] += 1
        
        try:
            # 2D 매핑이 활성화되어 있을 때는 매핑 성공 시에만 전송
            if self.enable_2d_mapping and self.screen_calibrator.is_calibrated:
                # 2D 매핑 데이터가 없으면 전송하지 않음 (오검출 방지)
                if not (laser_result.screen_coordinate and laser_result.unity_coordinate):
                    if laser_result.detected:
                        print(f"⚠️ 레이저 검출됐지만 캘리브레이션 영역 밖 - Unity 전송 스킵")
                        print(f"   위치: {laser_result.position}, 신뢰도: {laser_result.confidence:.2f}")
                        print(f"   → 레이저를 스크린 내부로 조준하세요!")
                    return False
            
            # Unity가 기대하는 JSON 형식으로 데이터 구성
            unity_data = {
                "detected": laser_result.detected,
                "confidence": laser_result.confidence,
                "position": [laser_result.position[0], laser_result.position[1]],  # Vector2
                "hsv_values": [laser_result.hsv_values[0], laser_result.hsv_values[1], laser_result.hsv_values[2]],  # Vector3
                "brightness": laser_result.brightness,
                "detection_method": laser_result.detection_method,
                "detection_time_ms": laser_result.detection_time_ms
            }
            
            # 2D 매핑 데이터 추가
            if laser_result.screen_coordinate and laser_result.unity_coordinate:
                unity_data["screen_coordinate"] = [laser_result.screen_coordinate[0], laser_result.screen_coordinate[1]]
                unity_data["unity_coordinate"] = [laser_result.unity_coordinate[0], laser_result.unity_coordinate[1]]
                
                # 성공 로그 (디버그용)
                print(f"📤 Unity 전송: 스크린({laser_result.screen_coordinate[0]:.3f}, {laser_result.screen_coordinate[1]:.3f}) "
                      f"→ Unity({laser_result.unity_coordinate[0]:.3f}, {laser_result.unity_coordinate[1]:.3f})")
            else:
                # 2D 매핑이 비활성화된 경우에만 기본값 전송
                unity_data["screen_coordinate"] = [0.0, 0.0]
                unity_data["unity_coordinate"] = [0.0, 0.0]
            
            # 스크린 크기 정보 추가 (캘리브레이션되어 있으면)
            if self.screen_calibrator.is_calibrated:
                unity_data["screen_real_size"] = {
                    "width": self.screen_calibrator.calibration_data.screen_real_size['width'],
                    "height": self.screen_calibrator.calibration_data.screen_real_size['height']
                }
            
            # JSON 직렬화 및 전송
            json_data = json.dumps(unity_data)
            
            # 1. Unity로 전송 (기존)
            self.udp_socket.sendto(json_data.encode('utf-8'), (self.unity_ip, self.unity_port))
            
            # 2. Vector Fusion System으로도 전송 (추가)
            if self.enable_vector_fusion and laser_result.screen_coordinate and laser_result.unity_coordinate:
                # Vector Fusion용 데이터 구성 (스크린 좌표 포함)
                vector_fusion_data = {
                    "detected": laser_result.detected,
                    "timestamp": time.time(),
                    "screen_position": [laser_result.screen_coordinate[0], laser_result.screen_coordinate[1]],
                    "confidence": laser_result.confidence,
                    "detection_method": laser_result.detection_method,
                    "camera_source": "modular",
                    "screen_size_mm": [2400.0, 1500.0]  # 실측 스크린 크기 (2.4m x 1.5m)
                }
                
                # Vector Fusion System으로 전송
                vector_json = json.dumps(vector_fusion_data)
                self.udp_socket.sendto(vector_json.encode('utf-8'), (self.vector_fusion_ip, self.vector_fusion_port))
            
            self.communication_stats['successful_sent'] += 1
            self.communication_stats['last_send_time'] = time.time()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Unity 전송 실패: {e}")
            self.communication_stats['failed_sent'] += 1
            return False
    
    def toggle_2d_mapping(self) -> bool:
        """2D 매핑 활성화/비활성화 토글"""
        if not self.screen_calibrator.is_calibrated:
            print("❌ 캘리브레이션이 완료되지 않았습니다. [K] 키로 캘리브레이션을 먼저 진행하세요.")
            return False
        
        self.enable_2d_mapping = not self.enable_2d_mapping
        status = "활성화" if self.enable_2d_mapping else "비활성화"
        print(f"🗺️ 2D 매핑: {status}")
        
        if self.enable_2d_mapping:
            accuracy = self.screen_calibrator.calibration_data.calibration_accuracy
            print(f"📊 현재 캘리브레이션 정확도: {accuracy*100:.1f}%")
        
        return self.enable_2d_mapping
    
    def get_mapping_status(self) -> Dict:
        """매핑 시스템 상태 반환"""
        mapping_status = "OFF"
        
        if self.screen_calibrator.is_calibrated:
            if self.enable_2d_mapping:
                accuracy = self.screen_calibrator.calibration_data.calibration_accuracy
                mapping_status = f"ON ({accuracy*100:.1f}%)"
            else:
                mapping_status = "Ready"
        elif self.calibration_mode:
            mapping_status = "Calibrating..."
        
        return {
            'status': mapping_status,
            'calibrated': self.screen_calibrator.is_calibrated,
            'enabled': self.enable_2d_mapping,
            'calibration_mode': self.calibration_mode,
            'accuracy': self.screen_calibrator.calibration_data.calibration_accuracy if self.screen_calibrator.is_calibrated else 0.0
        }
    
    def get_communication_status(self) -> Dict:
        """Unity 통신 상태 반환"""
        return {
            'enabled': self.enable_unity_communication,
            'connected': self.udp_socket is not None,
            'ip': self.unity_ip,
            'port': self.unity_port,
            'stats': self.communication_stats.copy()
        }
    
    def get_mapping_stats(self) -> Dict:
        """매핑 통계 반환"""
        success_rate = 0.0
        if self.mapping_stats['total_attempts'] > 0:
            success_rate = (self.mapping_stats['successful_mappings'] / self.mapping_stats['total_attempts']) * 100
        
        return {
            'total_attempts': self.mapping_stats['total_attempts'],
            'successful_mappings': self.mapping_stats['successful_mappings'],
            'out_of_bounds': self.mapping_stats['out_of_bounds'],
            'success_rate': success_rate,
            'calibration_accuracy': self.mapping_stats['calibration_accuracy']
        }
    
    def draw_calibration_overlay(self, display_image: np.ndarray) -> np.ndarray:
        """캘리브레이션 오버레이 그리기"""
        if not self.screen_calibrator.is_calibrated:
            return display_image
        
        try:
            # 캘리브레이션 영역 그리기 (초록색 사각형)
            calib_data = self.screen_calibrator.calibration_data
            corners = calib_data.screen_corners_pixel
            
            if len(corners) == 4:
                # 4개 모서리를 연결하는 선 그리기
                pts = np.array(corners, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # 초록색 테두리
                cv2.polylines(display_image, [pts], True, (0, 255, 0), 2)
                
                # 반투명 초록색 영역
                overlay = display_image.copy()
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.1, display_image, 0.9, 0, display_image)
                
                # 캘리브레이션 정보 표시
                accuracy = calib_data.calibration_accuracy
                cv2.putText(display_image, f"2D Mapping: {accuracy*100:.1f}%", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # 모서리 점 표시
                for i, (x, y) in enumerate(corners):
                    cv2.circle(display_image, (int(x), int(y)), 8, (0, 255, 0), -1)
                    cv2.putText(display_image, str(i+1), (int(x)-5, int(y)+5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        except Exception as e:
            self.logger.error(f"캘리브레이션 오버레이 그리기 실패: {e}")
        
        return display_image
    
    def print_detailed_stats(self):
        """상세 통계 출력"""
        print(f"\n🗺️ 2D 스크린 매핑 상세 통계:")
        
        # 매핑 상태
        status = self.get_mapping_status()
        print(f"  매핑 상태: {status['status']}")
        print(f"  캘리브레이션 완료: {'예' if status['calibrated'] else '아니오'}")
        print(f"  매핑 활성화: {'예' if status['enabled'] else '아니오'}")
        
        # 매핑 통계
        mapping_stats = self.get_mapping_stats()
        print(f"  매핑 시도: {mapping_stats['total_attempts']}회")
        print(f"  성공: {mapping_stats['successful_mappings']}회")
        print(f"  영역 밖: {mapping_stats['out_of_bounds']}회")
        print(f"  성공률: {mapping_stats['success_rate']:.1f}%")
        
        # Unity 통신 상태
        comm_status = self.get_communication_status()
        print(f"\n🌐 Unity 통신 상태:")
        print(f"  활성화: {'예' if comm_status['enabled'] else '아니오'}")
        print(f"  연결: {'예' if comm_status['connected'] else '아니오'}")
        print(f"  주소: {comm_status['ip']}:{comm_status['port']}")
        
        # 통신 통계
        stats = comm_status['stats']
        print(f"  전송 시도: {stats['total_sent']}회")
        print(f"  성공: {stats['successful_sent']}회")
        print(f"  실패: {stats['failed_sent']}회")
        
        if stats['total_sent'] > 0:
            success_rate = (stats['successful_sent'] / stats['total_sent']) * 100
            print(f"  통신 성공률: {success_rate:.1f}%")
    
    def cleanup(self):
        """리소스 정리"""
        if self.udp_socket:
            self.udp_socket.close()
            print("🌐 Unity UDP 통신 종료")
        
        print("🗺️ 스크린 매핑 모듈 종료")
    
    def is_calibrated(self) -> bool:
        """캘리브레이션 완료 여부 반환"""
        return self.screen_calibrator.is_calibrated
    
    def is_mapping_enabled(self) -> bool:
        """매핑 활성화 여부 반환"""
        return self.enable_2d_mapping
    
    def is_calibration_mode(self) -> bool:
        """캘리브레이션 모드 여부 반환"""
        return self.calibration_mode
    
    def get_calibration_progress(self) -> Dict:
        """캘리브레이션 진행 상황 반환"""
        if not self.calibration_mode:
            return {'active': False, 'progress': 0, 'next_step': None}
        
        points_added = len(self.screen_calibrator.calibration_data.screen_corners_pixel)
        
        steps = ['좌측상단', '우측상단', '우측하단', '좌측하단']
        next_step = steps[points_added] if points_added < 4 else None
        
        return {
            'active': True,
            'progress': points_added,
            'total_steps': 4,
            'next_step': next_step,
            'percentage': (points_added / 4) * 100
        } 