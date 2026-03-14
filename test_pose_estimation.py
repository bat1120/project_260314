import argparse
import urllib.request
import os
import cv2
try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics 패키지가 설치되어 있지 않습니다. 아래 명령어로 설치해주세요:")
    print("pip install ultralytics")
    exit(1)

def download_default_image(filename="default_person.jpg"):
    # 사람 이미지가 포함된 기본 이미지 다운로드 (제공되는 기본 예시 이미지)
    url = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/zidane.jpg"
    if not os.path.exists(filename):
        print(f"기본 이미지를 다운로드 중입니다: {url}")
        try:
            urllib.request.urlretrieve(url, filename)
            print("다운로드 완료.")
        except Exception as e:
            print(f"이미지 다운로드 실패: {e}")
            return None
    return filename

def test_pose_estimation(image_path):
    if not os.path.exists(image_path):
        print(f"오류: '{image_path}' 이미지를 찾을 수 없습니다.")
        return

    print("가벼운 Pose Estimation 모델(YOLOv8 nano pose)을 불러옵니다...")
    # 가장 가벼운 모델인 yolov8n-pose.pt 사용 (처음 실행 시 자동 다운로드 됨)
    model = YOLO('yolov8n-pose.pt') 

    print(f"'{image_path}' 이미지에 대한 추론을 시작합니다...")
    results = model(image_path)

    # 시각화 및 결과 저장
    for r in results:
        # plot() 메서드는 추론 결과(바운딩 박스 및 키포인트)가 그려진 numpy 배열(BGR)을 반환합니다.
        im_array = r.plot()
        
        output_filename = f"pose_result_{os.path.basename(image_path)}"
        cv2.imwrite(output_filename, im_array)
        print(f"시각화 결과가 '{output_filename}' 파일로 저장되었습니다.")
        
        # 화면에 결과 출력 (GUI 환경에서 작동)
        cv2.imshow("Pose Estimation Result", im_array)
        print("결과 창을 닫으려면 아무 키나 누르세요...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="가벼운 Pose Estimation 모델 테스트 스크립트")
    parser.add_argument('--image', type=str, default=None, help="테스트할 이미지 경로 (입력하지 않으면 기본 이미지 자동 다운로드)")
    args = parser.parse_args()

    img_path = args.image
    if img_path is None:
        print("입력된 이미지가 없습니다. 기본 이미지를 사용합니다.")
        img_path = download_default_image()
    
    if img_path:
        test_pose_estimation(img_path)
