"""
Auto-Pilot 자동 분류기 검증 스크립트
무작위 동물 5장, 식물 5장, 음식 5장을 테스트하여 대분류/소분류 정확도를 표로 출력합니다.
"""

import os
import sys
import random

# Windows 인코딩 이슈
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from image_classifier import load_clip_model, classify_image_auto, PRESET_CATEGORIES

def get_random_images(category, count=5):
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images", "accuracy_test", category)
    if not os.path.exists(base_dir):
        return []
    
    files = [f for f in os.listdir(base_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(files)
    
    # 5개만 선택하면서 각 이미지에 대한 "예상 정답(소분류 앞단어 영어)"를 파일명에서 추출
    # 예: "dog_01.jpg" -> "dog"
    selected = []
    for f in files[:count]:
        expected_fine = f.split('_')[0] 
        selected.append((os.path.join(base_dir, f), expected_fine))
    
    return selected

def check_correctness(result, expected_broad, expected_fine):
    if not result or result.get("mode") != "auto":
        return False, False
        
    broad_pred = result["broad_prediction"]["key"]  # 'animal', 'food', 'plant'
    fine_pred = result["top_prediction"]["class"].lower() # '개 (dog)' -> 'dog' 있는 지 확인
    
    broad_correct = (broad_pred == expected_broad)
    fine_correct = (expected_fine in fine_pred)
    
    return broad_correct, fine_correct

def main():
    print("🚀 Auto-Pilot 분류기 무작위 15장 정확도 벤치마크")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_clip_model(device)
    
    test_cases = [
        ("animal", get_random_images("animal", 5)),
        ("food", get_random_images("food", 5)),
        ("plant", get_random_images("plant", 5))
    ]
    
    results_table = []
    
    total_broad = 0
    total_fine = 0
    total_cnt = 0
    
    for category_key, images in test_cases:
        for idx, (img_path, expected_fine) in enumerate(images, 1):
            filename = os.path.basename(img_path)
            print(f"[*] 테스트 중 ({category_key}): {filename} ...")
            
            result = classify_image_auto(model, processor, img_path, device, lang="en")
            if not result:
                continue
                
            b_correct, f_correct = check_correctness(result, category_key, expected_fine)
            
            b_pred_name = result["broad_prediction"]["class"]
            f_pred_name = result["top_prediction"]["class"]
            conf = result["top_prediction"]["confidence"]
            
            # 결과 표에 넣을 한 줄
            status = "✅" if (b_correct and f_correct) else "❌"
            
            results_table.append({
                "category": PRESET_CATEGORIES[category_key]["name_ko"],
                "file": filename,
                "expected": expected_fine,
                "pred_broad": b_pred_name,
                "pred_fine": f_pred_name,
                "conf": f"{conf:.1f}%",
                "status": status
            })
            
            if b_correct: total_broad += 1
            if f_correct: total_fine += 1
            total_cnt += 1

    # 마크다운 표 출력
    print("\n\n" + "="*80)
    print("📊 Auto-Pilot 동작 결과 표")
    print("="*80)
    print(f"| {'테마 (Theme)':<15} | {'파일 (File)':<20} | {'대분류 예측':<15} | {'상세 사물 예측':<25} | {'정답 여부':<10} |")
    print(f"|{'-'*17}|{'-'*22}|{'-'*17}|{'-'*27}|{'-'*12}|")
    
    for row in results_table:
        print(f"| {row['category']:<15} | {row['file']:<20} | {row['pred_broad']:<15} | {row['pred_fine']} ({row['conf']}) {' '*(17-len(row['pred_fine']))}|    {row['status']}     |")
        
    print("="*80)
    print(f"🏆 대분류(주제) 정확도: {total_broad}/{total_cnt} ({(total_broad / total_cnt * 100):.1f}%)")
    print(f"🎯 소분류(사물) 정확도: {total_fine}/{total_cnt} ({(total_fine / total_cnt * 100):.1f}%)")
    print("="*80)

if __name__ == "__main__":
    main()
