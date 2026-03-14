import os
import re
import uuid
import sys
import traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Encode fix for Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="AI Interactive Coach API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
STATIC_DIR = Path("static")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Mount static directories
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# ==========================================
# Load AI Models
# ==========================================
pose_model = None
try:
    from ultralytics import YOLO
    print("Loading Pose Model (YOLOv8)...")
    pose_model = YOLO('yolov8n-pose.pt')
except Exception as e:
    print(f"Failed to load Pose Model: {e}")

reader = None
import torch
try:
    import easyocr
    print("Loading OCR Model (EasyOCR)...")
    reader = easyocr.Reader(['ko', 'en'], gpu=torch.cuda.is_available())
except Exception as e:
    print(f"Failed to load OCR Model: {e}")

try:
    from test_face_recognition import compare_faces
except Exception as e:
    print(f"Failed to load Face Recognition module: {e}")

import math

# ==========================================
# OCR Helper Functions (inlined from test_ocr.py)
# ==========================================
def is_english(text):
    alpha = re.findall(r'[a-zA-Z]', text)
    hangul = re.findall(r'[가-힣]', text)
    return len(alpha) > len(hangul) and len(alpha) >= 2

def contextual_translate_safe(texts_to_translate):
    """Batched version of translation for better performance."""
    import translators as ts
    if not texts_to_translate:
        return {}

    # Group into chunks of 15 to avoid API length limits and provide progress
    chunk_size = 15
    translated_map = {}
    delimiter = " || "
    
    for i in range(0, len(texts_to_translate), chunk_size):
        chunk = texts_to_translate[i : i + chunk_size]
        combined_text = delimiter.join(chunk)
        
        try:
            # Join and translate in one go
            translated_blob = ts.translate_text(combined_text, to_language='ko', translator='google')
            translated_pieces = translated_blob.split(delimiter)
            
            # Map back
            for original, translated in zip(chunk, translated_pieces):
                translated_map[original] = translated.strip()
        except Exception as e:
            print(f"Translation chunk error: {e}")
            # Fallback to individual for this chunk
            for original in chunk:
                try:
                    res = ts.translate_text(original, to_language='ko', translator='google')
                    translated_map[original] = res.strip()
                except:
                    translated_map[original] = original

    return translated_map

def ocr_visualize(image_path, results, translation_map, font_path="malgun.ttf"):
    # ... (remains same or I can skip since we use table now)
    return None

# ==========================================
# Pose Analysis Helper
# ==========================================
def calculate_angle(a, b, c):
    """Calculate angle at b given points a, b, c."""
    try:
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    except:
        return None

def analyze_posture(keypoints):
    """
    Keypoints mapping (YOLOv8):
    5,6: shoulders, 11,12: hips, 13,14: knees, 15,16: ankles
    """
    feedback = []
    
    # Extract coordinates (if confidence > 0.5)
    def get_kp(idx):
        kp = keypoints[idx]
        if kp[2] > 0.5: return (kp[0], kp[1])
        return None

    l_sh, r_sh = get_kp(5), get_kp(6)
    l_hp, r_hp = get_kp(11), get_kp(12)
    l_kn, r_kn = get_kp(13), get_kp(14)
    l_ak, r_ak = get_kp(15), get_kp(16)

    # 1. Squat Analysis
    if l_hp and l_kn and l_ak:
        knee_angle = calculate_angle(l_hp, l_kn, l_ak)
        if knee_angle:
            if knee_angle > 140:
                feedback.append("더 깊게 앉아주세요. (스쿼트 깊이 부족)")
            elif knee_angle < 70:
                feedback.append("너무 깊게 앉았습니다. 무릎 부상에 유의하세요.")
            else:
                feedback.append("스쿼트 깊이가 적절합니다. 👍")
    
    # 2. Back Angle Analysis (Shoulder-Hip horizontal check)
    if l_sh and l_hp:
        # Check if back is too vertical or too horizontal
        # Simplified: check dy/dx
        dx = abs(l_sh[0] - l_hp[0])
        dy = abs(l_sh[1] - l_hp[1])
        if dx > dy:
            feedback.append("허리를 조금 더 세워주세요.")
        else:
            feedback.append("허리 각도가 안정적입니다.")

    if not feedback:
        feedback.append("전신이 보이도록 서주시면 더 가이드가 정확해집니다.")
    
    return " | ".join(feedback)

# ==========================================
# Helper Functions
# ==========================================
def save_upload_file(upload_file: UploadFile) -> str:
    ext = os.path.splitext(upload_file.filename)[1]
    if not ext:
        ext = ".jpg"
    filename = f"{uuid.uuid4()}{ext}"
    filepath = OUTPUTS_DIR / filename
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    return str(filepath)

# ==========================================
# API Endpoints
# ==========================================

@app.post("/api/pose")
async def process_pose(file: UploadFile = File(...)):
    if pose_model is None:
        return JSONResponse(status_code=500, content={"error": "Pose model not loaded"})
    
    filepath = save_upload_file(file)
    try:
        results = pose_model(filepath)
        out_filename = f"out_pose_{Path(filepath).name}"
        out_filepath = str(OUTPUTS_DIR / out_filename)
        
        feedback = "분석 중..."
        if len(results) > 0:
            res_plotted = results[0].plot() # BGR array
            cv2.imwrite(out_filepath, res_plotted)
            
            # Keypoints analysis
            if hasattr(results[0], 'keypoints') and results[0].keypoints is not None:
                # results[0].keypoints.data is [N, 17, 3] (x, y, conf)
                kp_data = results[0].keypoints.data[0].cpu().numpy()
                feedback = analyze_posture(kp_data)
            
            return {
                "status": "success", 
                "result_url": f"/outputs/{out_filename}",
                "feedback": feedback
            }
        else:
            return {"status": "no_detection", "result_url": f"/outputs/{Path(filepath).name}", "feedback": "사람을 찾을 수 없습니다."}
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/face")
async def process_face(ref_file: UploadFile = File(...), cam_file: UploadFile = File(...)):
    ref_path = save_upload_file(ref_file)
    cam_path = save_upload_file(cam_file)
    
    try:
        similarity, is_same = compare_faces(ref_path, cam_path, threshold=0.6)
        if similarity is None:
             return {"status": "error", "message": "Face not detected in one or both images"}
        
        return {
            "status": "success",
            "similarity": float(similarity),
            "is_same": bool(is_same)
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/ocr")
async def process_ocr(file: UploadFile = File(...)):
    if reader is None:
        return JSONResponse(status_code=500, content={"error": "OCR model not loaded"})
    
    filepath = save_upload_file(file)
    try:
        print(f"[OCR] Processing: {filepath}")
        img_cv = cv2.imread(filepath)
        if img_cv is None:
            return JSONResponse(status_code=400, content={"error": "Cannot read uploaded image"})
        
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(img_gray)
        print(f"[OCR] Detected {len(results)} text regions")
        
        if not results:
             return {"status": "success", "data": [], "message": "No text detected"}
        
        english_texts = [text for (bbox, text, prob) in results if is_english(text)]
        translation_map = {}
        if english_texts:
            print(f"[OCR] Batched Translation for {len(english_texts)} texts...")
            translation_map = contextual_translate_safe(english_texts)
        
        # Prepare structured data for table
        ocr_data = []
        for (bbox, text, prob) in results:
            ocr_data.append({
                "original": text,
                "translated": translation_map.get(text, text),
                "confidence": round(float(prob), 4)
            })
        
        return {
            "status": "success",
            "data": ocr_data,
            "original_url": f"/outputs/{Path(filepath).name}"
        }
             
    except Exception as e:
        print(f"[OCR] Error: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def read_root():
    # Helper redirect to static/index.html
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")
