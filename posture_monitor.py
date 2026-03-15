import cv2
import mediapipe as mp
import time
import winsound
import tkinter as tk
from tkinter import messagebox
import threading
import numpy as np

# MediaPipe Tasks API 설정
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
    running_mode=VisionRunningMode.VIDEO)

# 전역 변수
baseline_shoulder_width_ratio = None
baseline_ear_shoulder_z_ratio = None
baseline_y_diff = None
calibrated = False

bad_posture_start_time = None
WARNING_DELAY = 3.0  # N초 이상 안 좋은 자세 유지 시 경고 팝업 발생

def show_warning_popup(message):
    """별도의 스레드에서 팝업창을 띄우는 함수"""
    def _popup():
        root = tk.Tk()
        root.withdraw() # 메인 윈도우 숨김
        root.attributes('-topmost', True) # 항상 위에 표시
        messagebox.showwarning("자세 경고!", message, parent=root)
        root.destroy()
    threading.Thread(target=_popup, daemon=True).start()

def main():
    global baseline_shoulder_width_ratio, baseline_ear_shoulder_z_ratio, baseline_y_diff, calibrated, bad_posture_start_time

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    print("웹캠이 연결되었습니다. 바른 자세를 취한 후 화면을 클릭하고 'Enter' 키를 눌러 캘리브레이션을 진행하세요.")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("비디오 프레임을 읽을 수 없습니다.")
                continue

            image = cv2.flip(image, 1) # 거울 모드
            
            # MediaPipe Image 객체로 변환
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # 비디오 모드 처리 (타임스탬프 필요)
            frame_timestamp_ms = int(time.time() * 1000)
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            h, w, _ = image.shape
            
            status_text = "Good"
            color = (0, 255, 0)
            warning_msg = ""
            is_bad_posture = False

            if pose_landmarker_result.pose_landmarks:
                # 첫 번째 사람의 랜드마크 데이터 가져오기
                landmarks = pose_landmarker_result.pose_landmarks[0]
                
                # 시각화 (간단한 원 그리기)
                for lm in landmarks:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(image, (x, y), 2, (255, 0, 0), -1)
                
                # 주요 랜드마크 추출 (0: 코, 7: 왼쪽 귀, 8: 오른쪽 귀, 11: 왼쪽 어깨, 12: 오른쪽 어깨)
                nose = landmarks[0]
                l_ear = landmarks[7]
                r_ear = landmarks[8]
                l_shoulder = landmarks[11]
                r_shoulder = landmarks[12]

                # 1. 어깨 너비 계산 (화면 픽셀 대비 비율)
                shoulder_width_px = abs(l_shoulder.x - r_shoulder.x) * w
                shoulder_width_ratio = shoulder_width_px / w

                # 2. 귀와 어깨의 Z축 거리 차이 (거북목 판단용)
                avg_ear_z = (l_ear.z + r_ear.z) / 2
                avg_shoulder_z = (l_shoulder.z + r_shoulder.z) / 2
                ear_shoulder_z_diff = avg_ear_z - avg_shoulder_z

                # 3. 코와 양 어깨 중앙의 Y축 거리 (목이 짧아지는 것을 감지)
                avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
                nose_shoulder_y_diff = avg_shoulder_y - nose.y

                if not calibrated:
                    cv2.putText(image, "Press 'ENTER' to calibrate good posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # --- 거리 모니터링 (Too Close) ---
                    # 등록된 어깨 너비보다 1.3배 이상 넓게 보이면 가깝다고 판단
                    if shoulder_width_ratio > baseline_shoulder_width_ratio * 1.3:
                        status_text = "TOO CLOSE"
                        warning_msg = "모니터와 너무 가깝습니다! 뒤로 물러나세요."
                        color = (0, 0, 255)
                        is_bad_posture = True

                    # --- 거북목 모니터링 (Turtle Neck) ---
                    # 귀가 어깨보다 센서쪽으로 비정상적으로 많이 나오거나, 수직 거리가 비정상적으로 짧아지는 경우
                    elif ear_shoulder_z_diff < baseline_ear_shoulder_z_ratio - 0.05 or nose_shoulder_y_diff < baseline_y_diff * 0.8:
                        status_text = "TURTLE NECK"
                        warning_msg = "거북목 자세입니다! 허리와 목을 곧게 펴세요."
                        color = (0, 0, 255)
                        is_bad_posture = True
                    
                    # 경고 지속 시간 체크 및 알림 발생
                    if is_bad_posture:
                        if bad_posture_start_time is None:
                            bad_posture_start_time = time.time()
                        elif time.time() - bad_posture_start_time > WARNING_DELAY:
                            winsound.Beep(1000, 500) # 주파수 1000Hz, 0.5초 지속
                            show_warning_popup(warning_msg)
                            time.sleep(1) # 연속 팝업을 위한 최소 대기시간
                            bad_posture_start_time = None 
                    else:
                        bad_posture_start_time = None

                    cv2.putText(image, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    if warning_msg:
                        cv2.putText(image, "WARNING!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Posture Monitor", image)

            key = cv2.waitKey(5) & 0xFF
            if key == 13 and not calibrated and pose_landmarker_result.pose_landmarks: # Enter 키를 누르면 캘리브레이션
                baseline_shoulder_width_ratio = shoulder_width_ratio
                baseline_ear_shoulder_z_ratio = ear_shoulder_z_diff
                baseline_y_diff = nose_shoulder_y_diff
                calibrated = True
                print("기준 자세가 저장되었습니다. 모니터링을 시작합니다.")
                winsound.Beep(1500, 300)

            elif key == ord('q'): # 'q' 키를 누르면 종료
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
