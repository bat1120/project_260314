"""
AI Interactive Coach (홈트레이닝 코칭 플랫폼)
다양한 AI 모델을 통합하여 사용자에게 인터랙티브한 코칭과 분석을 제공합니다.
"""

import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
import os
import sys

# Windows 환경 인코딩 이슈 대비
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Pose Estimation
try:
    from ultralytics import YOLO
except ImportError:
    print("ultralytics 패키지가 필요합니다.")

# Face Recognition
try:
    from test_face_recognition import compare_faces
except ImportError:
    print("test_face_recognition.py 파일을 불러올 수 없습니다.")

# OCR
try:
    import easyocr
    from test_ocr import is_english, contextual_translate, visualize_and_save
except ImportError:
    print("easyocr 또는 test_ocr.py 파일을 불러올 수 없습니다.")

# 임시 파일 경로
TEMP_FACE_1 = "temp_face1.jpg"
TEMP_FACE_2 = "temp_face2.jpg"
TEMP_OCR = "temp_ocr.jpg"

# ==========================================
# 1. Pose Estimation (가상 PT)
# ==========================================
print("Loading Pose Model...")
try:
    pose_model = YOLO('yolov8n-pose.pt')
except:
    pose_model = None

def process_pose(image):
    if image is None:
        return None
    if pose_model is None:
        return image
        
    print("Processing Pose Estimation...")
    # YOLO 모델 추론 (Gradio의 RGB Numpy 배열을 그대로 사용 가능)
    results = pose_model(image)
    if len(results) > 0:
        res_plotted = results[0].plot() # BGR 배열 반환
        return cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    return image

# ==========================================
# 2. Face Recognition (회원 인증)
# ==========================================
def process_face(img1, img2):
    if img1 is None or img2 is None:
        return "두 이미지를 모두 업로드해주세요.", None
    
    print("Processing Face Recognition...")
    # 임시 파일로 저장 후 처리
    Image.fromarray(img1).save(TEMP_FACE_1)
    Image.fromarray(img2).save(TEMP_FACE_2)
    
    similarity, is_same = compare_faces(TEMP_FACE_1, TEMP_FACE_2, threshold=0.6)
    
    if similarity is None:
        return "얼굴을 인식할 수 없습니다. 다른 사진을 업로드해주세요.", None
        
    result_text = f"유사도: {similarity:.4f}\n동일인 여부: {'일치' if is_same else '불일치'}"
    status = "✅ 인증 성공" if is_same else "❌ 인증 실패"
    
    print(result_text)
    return result_text, status

# ==========================================
# 3. OCR & Translate (성분표 번역)
# ==========================================
print("Loading OCR Model...")
try:
    reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
except:
    reader = None

def process_ocr(image):
    if image is None:
        return None
    if reader is None:
        return image
        
    print("Processing OCR and Translation...")
    # 임시 파일로 저장 후 처리
    Image.fromarray(image).save(TEMP_OCR)
    
    results = reader.readtext(TEMP_OCR)
    if not results:
        print("No text detected.")
        return image
        
    english_texts = [text for (bbox, text, prob) in results if is_english(text)]
    translation_map = {}
    
    if english_texts:
        translation_map = contextual_translate(english_texts)
        
    result_img = visualize_and_save(TEMP_OCR, results, translation_map, font_path="malgun.ttf")
    
    if result_img:
        # PIL Image를 반환
        return result_img
    
    return image

# ==========================================
# Gradio UI 구성
# ==========================================
custom_css = """
.gradio-container {
    font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif;
}
.title-text {
    text-align: center;
    color: #2D3748;
    margin-bottom: 2rem;
}
"""

with gr.Blocks(title="AI Interactive Coach") as demo:
    gr.HTML("<h1 class='title-text'>🏃‍♂️ AI Interactive Coach (홈트레이닝 코칭 플랫폼)</h1>")
    gr.Markdown("### 웹캠이나 사진을 활용하여 자세 교정, 회원 인증, 보충제 성분 분석을 수행하는 통합 플랫폼입니다.")
    
    with gr.Tabs():
        # 1. 가상 PT 탭
        with gr.TabItem("🧘 가상 PT (자세 분석)"):
            gr.Markdown("운동하는 모습이나 웹캠을 통해 실시간으로 관절(Pose)을 추출하고 자세를 분석합니다.")
            with gr.Row():
                with gr.Column():
                    pose_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="입력 이미지 또는 웹캠")
                    pose_btn = gr.Button("자세 분석하기", variant="primary", size="lg")
                with gr.Column():
                    pose_output = gr.Image(type="numpy", label="분석 결과")
            
            pose_btn.click(fn=process_pose, inputs=pose_input, outputs=pose_output)
            
        # 2. 회원 인증 탭
        with gr.TabItem("👤 회원 인증 (Face Recognition)"):
            gr.Markdown("등록된 프로필 사진과 현재 촬영한 사진을 비교하여 본인 인증을 수행합니다.")
            with gr.Row():
                with gr.Column():
                    face_ref = gr.Image(sources=["upload"], type="numpy", label="프로필 사진 (기준)")
                    face_webcam = gr.Image(sources=["upload", "webcam"], type="numpy", label="현재 사진 (웹캠/업로드)")
                    face_btn = gr.Button("본인 인증하기", variant="primary", size="lg")
                with gr.Column():
                    face_status = gr.Textbox(label="인증 결과")
                    face_result_text = gr.Textbox(label="상세 정보 (유사도 점수)", lines=2)
            
            face_btn.click(fn=process_face, inputs=[face_ref, face_webcam], outputs=[face_result_text, face_status])
            
        # 3. 보충제 성분 번역 탭
        with gr.TabItem("💊 성분표 번역 (OCR + Translate)"):
            gr.Markdown("해외 보충제나 영양제 성분표를 촬영하면 텍스트를 인식하고 한국어로 자동 번역하여 보여드립니다.")
            with gr.Row():
                with gr.Column():
                    ocr_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="성분표 이미지")
                    ocr_btn = gr.Button("인식 및 번역하기", variant="primary", size="lg")
                with gr.Column():
                    ocr_output = gr.Image(type="pil", label="번역 결과")
                    
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, css=custom_css, theme=gr.themes.Soft())
