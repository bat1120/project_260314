import os
import sys

# Windows 인코딩 이슈
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from image_classifier import load_clip_model, classify_image_auto, PRESET_CATEGORIES

def get_images(folder):
    if not os.path.exists(folder): return []
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]

def main():
    folders = {
        "animal": r"c:\Users\804\Desktop\동물",
        "plant": r"c:\Users\804\Desktop\식물",
        "food": r"c:\Users\804\Desktop\음식"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, processor = load_clip_model(device)

    results_table = []
    total = 0
    broad_correct = 0

    for category_key, folder_path in folders.items():
        images = get_images(folder_path)
        for img_path in images:
            filename = os.path.basename(img_path)
            
            result = classify_image_auto(model, processor, img_path, device, lang="ko")
            if not result:
                continue

            b_pred_key = result["broad_prediction"]["key"]
            b_pred_name = result["broad_prediction"]["class"]
            f_pred_name = result["top_prediction"]["class"]
            conf = result["top_prediction"]["confidence"]
            
            is_broad_correct = (b_pred_key == category_key)
            if is_broad_correct:
                broad_correct += 1
            total += 1

            results_table.append({
                "folder": os.path.basename(folder_path),
                "file": filename,
                "pred_broad": b_pred_name,
                "pred_fine": f_pred_name,
                "conf": f"{conf:.1f}%",
                "is_broad_correct": is_broad_correct
            })
    
    print("\n\n" + "="*80)
    print("📊 바탕화면 폴더 안의 이미지 인식 결과 표")
    print("="*80)
    print(f"| {'폴더 (Folder)':<10} | {'파일명 (File)':<40} | {'대분류 예측':<15} | {'상세 예측 (소분류)':<20} | {'대분류 일치':<8} |")
    print(f"|{'-'*12}|{'-'*42}|{'-'*17}|{'-'*24}|{'-'*11}|")
    
    for row in results_table:
        status = "✅" if row['is_broad_correct'] else "❌"
        print(f"| {row['folder']:<10} | {row['file'][:40]:<40} | {row['pred_broad']:<15} | {row['pred_fine']} ({row['conf']}) {' '*(12-len(row['pred_fine']))}|    {status}   |")
        
    print("="*80)
    if total > 0:
        print(f"🏆 대분류 기준 판단 정확도: {broad_correct}/{total} ({(broad_correct / total * 100):.1f}%)")
        print("💡 소분류(개, 스시, 장미 등)에 대한 정답은 파일명으로 판단하기 어려워 모델의 예측값만 출력하였습니다.")
    print("="*80)

if __name__ == "__main__":
    main()
