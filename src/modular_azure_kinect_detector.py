#!/usr/bin/env python3
"""
Virtra 모듈화된 Azure Kinect DK 레이저 검출 시스템
기존 모듈화 구조를 유지하면서 Azure Kinect DK 지원

모듈 구성 (기존 그대로 유지):
1. laser_detector_core.py: 핵심 검출 알고리즘 + HSV 학습
2. frame_processing.py: 프레임 차이 검출 + 모션 필터링
3. screen_mapping.py: 2D 스크린 매핑 + Unity 통신
4. camera_manager.py: 통합 카메라 관리 (신규)
5. main_controller_azure_kinect.py: Azure Kinect 통합 컨트롤러 (신규)

지원 카메라:
- 웹캠 (기존)
- ZED (기존)
- Azure Kinect DK 단일 (신규)
- Azure Kinect DK 듀얼 (신규)
"""

import sys
import logging
import argparse
from typing import Dict, Any
from main_controller_azure_kinect import MainControllerAzureKinect

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('modular_azure_kinect_detector.log')
        ]
    )

def parse_arguments():
    """명령행 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="Virtra 모듈화된 Azure Kinect DK 레이저 검출 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python modular_azure_kinect_detector.py                          # 기본 Azure Kinect HD 모드
  python modular_azure_kinect_detector.py --camera azure_kinect    # Azure Kinect 명시적 지정
  python modular_azure_kinect_detector.py --camera azure_kinect --4k  # Azure Kinect 4K 모드
  python modular_azure_kinect_detector.py --camera dual_azure_kinect  # 듀얼 Azure Kinect
  python modular_azure_kinect_detector.py --camera dual_azure_kinect --4k  # 듀얼 4K 모드
  python modular_azure_kinect_detector.py --enhanced              # Enhanced Detector 활성화
  python modular_azure_kinect_detector.py --enhanced --enable-cht # Enhanced + CHT 즉시 활성화
  python modular_azure_kinect_detector.py --camera webcam         # 웹캠 (기존 호환성)
  python modular_azure_kinect_detector.py --camera zed            # ZED (기존 호환성)
        """
    )
    
    parser.add_argument(
        '--camera', 
        choices=['webcam', 'zed', 'azure_kinect', 'dual_azure_kinect'],
        default='azure_kinect',
        help='카메라 타입 선택 (기본: azure_kinect)'
    )
    
    parser.add_argument(
        '--device-id',
        type=int,
        default=0,
        help='카메라 디바이스 ID (기본: 0)'
    )
    
    parser.add_argument(
        '--4k',
        action='store_true',
        dest='use_4k',
        help='Azure Kinect 4K 모드 활성화'
    )
    
    parser.add_argument(
        '--no-depth',
        action='store_true',
        help='깊이 센서 비활성화'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='디버그 모드 활성화'
    )
    
    parser.add_argument(
        '--enable-body-tracking',
        action='store_true',
        dest='enable_bt',
        help='Azure Kinect Body Tracking 활성화 (기본: 비활성)'
    )
    
    parser.add_argument(
        '--enhanced',
        action='store_true',
        help='Enhanced Detector 활성화 (Modified CHT 포함)'
    )
    
    parser.add_argument(
        '--enable-cht',
        action='store_true',
        help='Modified Circular Hough Transform 즉시 활성화'
    )
    
    return parser.parse_args()

def print_system_info(camera_type: str, **kwargs):
    """시스템 정보 출력"""
    print("Virtra Azure Kinect DK Laser Detection System")
    print("="*80)
    print("Module Configuration:")
    print("   laser_detector_core.py: Core detection + HSV learning")
    print("   frame_processing.py: Frame difference + motion filtering")
    print("   screen_mapping.py: 2D screen mapping + Unity communication")
    print("   camera_manager.py: Unified camera management")
    print("   main_controller_azure_kinect.py: Azure Kinect controller")
    if kwargs.get('use_enhanced_detector', False):
        print("   enhanced_laser_detector.py: Modified CHT + enhanced detection")
    
    # Enhanced Mode 정보 표시
    if kwargs.get('use_enhanced_detector', False):
        print()
        print("Enhanced Mode:")
        print(f"   Modified CHT: {'Enabled' if kwargs.get('enable_cht', False) else 'Runtime Toggle (H key)'}")
        print("   False Positive Reduction: Modified Circular Hough Transform")
        print("   Real-time Toggle: H key for CHT ON/OFF")
        print("   Performance Monitoring: E key for statistics")
    print()
    print(f"Current Configuration:")
    print(f"   Camera: {camera_type.upper()}")
    
    if camera_type in ['azure_kinect', 'dual_azure_kinect']:
        mode = "4K" if kwargs.get('use_4k', False) else "HD"
        print(f"   Resolution: {mode} mode")
        depth = "Disabled" if kwargs.get('no_depth', False) else "Enabled"
        print(f"   Depth Sensor: {depth}")
        
        if camera_type == 'dual_azure_kinect':
            print(f"   Config: Dual Camera (3D Triangulation)")
        else:
            print(f"   Config: Single Camera")
    
    elif camera_type == 'zed':
        print(f"   Resolution: HD 720p")
        print(f"   Depth Sensor: Stereo Vision")
    
    elif camera_type == 'webcam':
        print(f"   Resolution: 640x480")
        print(f"   Depth Sensor: None")
    
    print()
    print("Key Features:")
    if camera_type in ['azure_kinect', 'dual_azure_kinect']:
        print("   - 4K RGB Camera Support")
        print("   - TOF Depth Sensor (±2mm accuracy)")
        print("   - Real-time 3D Coordinate Calculation")
        print("   - Depth-based Laser Filtering")
        print("   - Sub-pixel Accuracy")
        if camera_type == 'dual_azure_kinect':
            print("   - 3D Triangulation")
    
    print("   - Modular Architecture")
    print("   - Unity Real-time Communication")
    print("   - 2D Screen Mapping")
    print("   - HSV Learning System")
    print("="*80)

def get_unity_port_info(camera_type: str) -> Dict[str, Any]:
    """Unity 포트 정보 반환"""
    port_mapping = {
        'webcam': {'port': 12345, 'description': '웹캠 전용'},
        'zed': {'port': 9998, 'description': 'ZED 전용'},
        'azure_kinect': {'port': 9999, 'description': 'Azure Kinect 단일'},
        'dual_azure_kinect': {'port': 9997, 'description': 'Azure Kinect 듀얼'}
    }
    return port_mapping.get(camera_type, port_mapping['webcam'])

def main():
    """메인 실행 함수"""
    # 명령행 인수 파싱
    args = parse_arguments()
    
    # 로깅 설정
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    setup_logging()
    
    # 카메라 설정 준비
    camera_kwargs = {
        'device_id': args.device_id,
        'use_4k': args.use_4k,
        'use_depth': not args.no_depth,
        'use_enhanced_detector': args.enhanced,  # Enhanced Detector 사용 여부
        'enable_cht': args.enable_cht,           # Modified CHT 즉시 활성화 여부
        'enable_bt': args.enable_bt              # Body Tracking 사용 여부
    }
    
    # 시스템 정보 출력
    print_system_info(args.camera, **camera_kwargs)
    
    # Unity 포트 정보
    unity_info = get_unity_port_info(args.camera)
    print(f"Unity Communication Port: {unity_info['port']} ({unity_info['description']})")
    print()
    
    # 카메라별 특별 안내
    if args.camera in ['azure_kinect', 'dual_azure_kinect']:
        print("Azure Kinect DK Requirements:")
        print("   1. Azure Kinect SDK Installation")
        print("   2. USB 3.0 Port Connection")
        print("   3. Power Adapter Required")
        if args.camera == 'dual_azure_kinect':
            print("   4. Second Kinect on Different USB Controller")
        print()
    
    try:
        # 메인 컨트롤러 초기화 및 실행
        print("Starting System...")
        controller = MainControllerAzureKinect(
            camera_type=args.camera,
            **camera_kwargs
        )
        
        # 검출 시작
        controller.run_detection()
        
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 중단됨")
    
    except ImportError as e:
        print(f"\n❌ 모듈 임포트 오류: {e}")
        print("\n💡 해결 방법:")
        
        if "pykinect" in str(e):
            print("   Azure Kinect 관련:")
            print("   1. pip install pykinect-azure")
            print("   2. Azure Kinect SDK 설치 확인")
            print("   3. 환경 변수 AZUREKINECT_SDK 설정")
        
        elif "pyzed" in str(e):
            print("   ZED 관련:")
            print("   1. ZED SDK 설치")
            print("   2. Python API 설치")
        
        else:
            print("   일반적인 해결책:")
            print("   1. 가상환경 활성화: venv\\Scripts\\activate")
            print("   2. 필요한 패키지 설치: pip install -r requirements.txt")
            print("   3. 모듈 파일 존재 확인")
        
        sys.exit(1)
    
    except Exception as e:
        print(f"\n[ERROR] Runtime Error: {e}")
        
        # 오류 타입별 안내
        if "camera" in str(e).lower():
            print("\n[INFO] Camera Issues:")
            print("   1. Check camera connection")
            print("   2. Close other camera applications")
            print("   3. Update drivers")
            if args.camera in ['azure_kinect', 'dual_azure_kinect']:
                print("   4. Check Azure Kinect power adapter")
                print("   5. Use USB 3.0 port")
        
        elif "unity" in str(e).lower():
            print("\n[INFO] Unity Communication Issues:")
            unity_info = get_unity_port_info(args.camera)
            print(f"   1. Unity port {unity_info['port']} configuration")
            print("   2. Check firewall settings")
            print("   3. Enable Unity scripts")
        
        if args.debug:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)

def show_available_systems():
    """사용 가능한 시스템 확인 및 표시"""
    print("🔍 사용 가능한 카메라 시스템 확인 중...")
    
    available_systems = []
    
    # 웹캠 확인
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            available_systems.append("webcam")
        cap.release()
    except:
        pass
    
    # ZED 확인
    try:
        import pyzed.sl as sl
        available_systems.append("zed")
    except ImportError:
        pass
    
    # Azure Kinect 확인
    try:
        import pykinect_azure as pykinect
        available_systems.extend(["azure_kinect", "dual_azure_kinect"])
    except ImportError:
        pass
    
    print("✅ 사용 가능한 시스템:")
    for system in available_systems:
        unity_info = get_unity_port_info(system)
        print(f"   📹 {system} (포트: {unity_info['port']})")
    
    if not available_systems:
        print("❌ 사용 가능한 카메라 시스템이 없습니다.")
    
    return available_systems

if __name__ == "__main__":
    # 도움말 옵션 처리
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        main()
    elif len(sys.argv) > 1 and sys.argv[1] == '--check':
        show_available_systems()
    else:
        main()