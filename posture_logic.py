import cv2
import mediapipe as mp
import time
import numpy as np

# MediaPipe Tasks API
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

_landmarker = None

def get_landmarker():
    global _landmarker
    if _landmarker is None:
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
            running_mode=VisionRunningMode.VIDEO)
        _landmarker = PoseLandmarker.create_from_options(options)
    return _landmarker

class PostureAnalyzer:
    def __init__(self):
        self.calibrated = False
        self.baseline_shoulder_width_ratio = None
        self.baseline_ear_shoulder_z_ratio = None
        self.baseline_y_diff = None

    def calibrate(self, shoulder_width_ratio, ear_shoulder_z_diff, nose_shoulder_y_diff):
        self.baseline_shoulder_width_ratio = shoulder_width_ratio
        self.baseline_ear_shoulder_z_ratio = ear_shoulder_z_diff
        self.baseline_y_diff = nose_shoulder_y_diff
        self.calibrated = True

    def analyze_frame(self, image_np):
        """
        image_np: RGB numpy array
        returns: (status_text, color_hex, warning_msg, is_bad_posture, landmarks_coords)
        """
        landmarker = get_landmarker()
        
        h, w, _ = image_np.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_np)
        frame_timestamp_ms = int(time.time() * 1000)
        
        try:
            pose_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        except Exception as e:
            print(f"MediaPipe Error: {e}")
            return "Error", "#ff0000", "분석 에러", False, []

        status_text = "Good"
        color_hex = "#22c55e" # Green
        warning_msg = ""
        is_bad_posture = False
        parsed_landmarks = []

        if pose_result and pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            
            # 클라이언트(JS)에 캔버스에 그릴 좌표 전달
            for idx, lm in enumerate(landmarks):
                # 코(0), 귀(7,8), 어깨(11,12) 표시용
                if idx in [0, 7, 8, 11, 12]:
                    parsed_landmarks.append({"x": lm.x, "y": lm.y})

            nose = landmarks[0]
            l_ear = landmarks[7]
            r_ear = landmarks[8]
            l_shoulder = landmarks[11]
            r_shoulder = landmarks[12]

            shoulder_width_px = abs(l_shoulder.x - r_shoulder.x) * w
            shoulder_width_ratio = shoulder_width_px / w

            avg_ear_z = (l_ear.z + r_ear.z) / 2
            avg_shoulder_z = (l_shoulder.z + r_shoulder.z) / 2
            ear_shoulder_z_diff = avg_ear_z - avg_shoulder_z

            avg_shoulder_y = (l_shoulder.y + r_shoulder.y) / 2
            nose_shoulder_y_diff = avg_shoulder_y - nose.y

            if not self.calibrated:
                status_text = "Not Calibrated"
                warning_msg = "화면 중앙에 맞추고 캘리브레이션 버튼을 누르세요."
                color_hex = "#eab308" # Yellow
                # 캘리브레이션 되지 않았을 때는 분석 결과값만 반환하여 클라이언트에서 임시 저장할 수 있게 함
                return status_text, color_hex, warning_msg, is_bad_posture, parsed_landmarks, {
                    "shoulder_width_ratio": shoulder_width_ratio,
                    "ear_shoulder_z_diff": ear_shoulder_z_diff,
                    "nose_shoulder_y_diff": nose_shoulder_y_diff
                }
            else:
                # --- 거리 모니터링 (Too Close) ---
                if shoulder_width_ratio > self.baseline_shoulder_width_ratio * 1.3:
                    status_text = "TOO CLOSE"
                    warning_msg = "모니터와 너무 가깝습니다! 뒤로 물러나세요."
                    color_hex = "#ef4444" # Red
                    is_bad_posture = True

                # --- 거북목 모니터링 (Turtle Neck) ---
                elif ear_shoulder_z_diff < self.baseline_ear_shoulder_z_ratio - 0.05 or nose_shoulder_y_diff < self.baseline_y_diff * 0.8:
                    status_text = "TURTLE NECK"
                    warning_msg = "거북목 자세입니다! 허리와 목을 곧게 펴세요."
                    color_hex = "#ef4444" # Red
                    is_bad_posture = True

                return status_text, color_hex, warning_msg, is_bad_posture, parsed_landmarks, None
        
        return "Not Detected", "#94a3b8", "사람을 찾을 수 없습니다.", False, [], None
