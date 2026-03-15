"""
이미지 분류기 정확도 테스트 스크립트
====================================
카테고리별로 웹에서 이미지를 다운로드하고 CLIP 분류기의 정확도를 측정합니다.

사용법:
    python test_classifier_accuracy.py
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from collections import defaultdict

# Windows 환경 인코딩 이슈 대비
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================================
# 테스트 이미지 URL 정의 (카테고리별 10~15장)
# Pexels, Unsplash, Pixabay 등 직접 이미지 URL 사용
# ============================================================================
TEST_IMAGES = {
    "animal": {
        "preset": "animal",
        "images": {
            # 개 (dog)
            "dog_01.jpg": {"url": "https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_640.jpg", "expected": "개 (dog)"},
            "dog_02.jpg": {"url": "https://cdn.pixabay.com/photo/2019/08/19/07/45/corgi-4415649_640.jpg", "expected": "개 (dog)"},
            "dog_03.jpg": {"url": "https://cdn.pixabay.com/photo/2016/02/19/15/46/labrador-retriever-1210559_640.jpg", "expected": "개 (dog)"},
            "dog_04.jpg": {"url": "https://cdn.pixabay.com/photo/2018/03/31/06/31/dog-3277416_640.jpg", "expected": "개 (dog)"},
            # 고양이 (cat)
            "cat_01.jpg": {"url": "https://cdn.pixabay.com/photo/2017/02/20/18/03/cat-2083492_640.jpg", "expected": "고양이 (cat)"},
            "cat_02.jpg": {"url": "https://cdn.pixabay.com/photo/2014/11/30/14/11/cat-551554_640.jpg", "expected": "고양이 (cat)"},
            "cat_03.jpg": {"url": "https://cdn.pixabay.com/photo/2017/11/09/21/41/cat-2934720_640.jpg", "expected": "고양이 (cat)"},
            # 새 (bird)
            "bird_01.jpg": {"url": "https://cdn.pixabay.com/photo/2017/02/07/16/47/kingfisher-2046453_640.jpg", "expected": "새 (bird)"},
            "bird_02.jpg": {"url": "https://cdn.pixabay.com/photo/2018/05/03/22/34/robin-3372529_640.jpg", "expected": "새 (bird)"},
            # 말 (horse)
            "horse_01.jpg": {"url": "https://cdn.pixabay.com/photo/2019/07/31/11/36/horse-4375484_640.jpg", "expected": "말 (horse)"},
            "horse_02.jpg": {"url": "https://cdn.pixabay.com/photo/2016/11/22/21/42/horse-1850250_640.jpg", "expected": "말 (horse)"},
            # 코끼리 (elephant)
            "elephant_01.jpg": {"url": "https://cdn.pixabay.com/photo/2016/11/14/04/45/elephant-1822636_640.jpg", "expected": "코끼리 (elephant)"},
            "elephant_02.jpg": {"url": "https://cdn.pixabay.com/photo/2013/09/03/19/35/african-bush-elephant-178999_640.jpg", "expected": "코끼리 (elephant)"},
            # 토끼 (rabbit)
            "rabbit_01.jpg": {"url": "https://cdn.pixabay.com/photo/2014/03/14/20/07/rabbit-287903_640.jpg", "expected": "토끼 (rabbit)"},
            # 펭귄 (penguin)
            "penguin_01.jpg": {"url": "https://cdn.pixabay.com/photo/2018/06/30/09/29/penguin-3507671_640.jpg", "expected": "펭귄 (penguin)"},
        }
    },
    "food": {
        "preset": "food",
        "images": {
            # 피자
            "pizza_01.jpg": {"url": "https://cdn.pixabay.com/photo/2017/12/09/08/18/pizza-3007395_640.jpg", "expected": "피자 (pizza)"},
            "pizza_02.jpg": {"url": "https://cdn.pixabay.com/photo/2017/01/03/11/33/pizza-1949183_640.jpg", "expected": "피자 (pizza)"},
            "pizza_03.jpg": {"url": "https://cdn.pixabay.com/photo/2016/06/08/00/03/pizza-1442946_640.jpg", "expected": "피자 (pizza)"},
            # 스시
            "sushi_01.jpg": {"url": "https://cdn.pixabay.com/photo/2017/10/15/11/41/sushi-2853382_640.jpg", "expected": "스시 (sushi)"},
            "sushi_02.jpg": {"url": "https://cdn.pixabay.com/photo/2020/04/04/15/07/sushi-5002639_640.jpg", "expected": "스시 (sushi)"},
            # 햄버거
            "burger_01.jpg": {"url": "https://cdn.pixabay.com/photo/2016/03/05/19/02/hamburger-1238246_640.jpg", "expected": "햄버거 (hamburger)"},
            "burger_02.jpg": {"url": "https://cdn.pixabay.com/photo/2014/10/19/20/59/hamburger-494706_640.jpg", "expected": "햄버거 (hamburger)"},
            # 파스타
            "pasta_01.jpg": {"url": "https://cdn.pixabay.com/photo/2018/07/18/19/12/pasta-3547078_640.jpg", "expected": "파스타 (pasta)"},
            "pasta_02.jpg": {"url": "https://cdn.pixabay.com/photo/2016/11/23/18/31/pasta-1854245_640.jpg", "expected": "파스타 (pasta)"},
            # 케이크
            "cake_01.jpg": {"url": "https://cdn.pixabay.com/photo/2017/01/11/11/33/cake-1971552_640.jpg", "expected": "케이크 (cake)"},
            "cake_02.jpg": {"url": "https://cdn.pixabay.com/photo/2014/06/11/17/00/cake-366346_640.jpg", "expected": "케이크 (cake)"},
            # 샐러드
            "salad_01.jpg": {"url": "https://cdn.pixabay.com/photo/2017/09/16/19/21/salad-2756467_640.jpg", "expected": "샐러드 (salad)"},
            "salad_02.jpg": {"url": "https://cdn.pixabay.com/photo/2015/05/31/13/59/salad-791891_640.jpg", "expected": "샐러드 (salad)"},
            # 아이스크림
            "icecream_01.jpg": {"url": "https://cdn.pixabay.com/photo/2016/09/29/13/08/ice-1702990_640.jpg", "expected": "아이스크림 (ice cream)"},
            # 스테이크
            "steak_01.jpg": {"url": "https://cdn.pixabay.com/photo/2016/01/22/02/13/meat-1155132_640.jpg", "expected": "스테이크 (steak)"},
        }
    },
    "plant": {
        "preset": "plant",
        "images": {
            # 장미 (Unsplash)
            "rose_01.jpg": {"url": "https://images.unsplash.com/photo-1586082207282-3dcb61d25ebd?q=80&w=640", "expected": "장미 (rose)"},
            "rose_02.jpg": {"url": "https://images.unsplash.com/photo-1703236042519-d0afaf4a9afc?w=640", "expected": "장미 (rose)"},
            "rose_03.jpg": {"url": "https://images.unsplash.com/photo-1736888410251-8ffedce63909?w=640", "expected": "장미 (rose)"},
            # 해바라기 (Unsplash)
            "sunflower_01.jpg": {"url": "https://images.unsplash.com/photo-1597848212624-a19eb35e2651?w=640", "expected": "해바라기 (sunflower)"},
            "sunflower_02.jpg": {"url": "https://images.unsplash.com/photo-1542801204-141ec23989d7?w=640", "expected": "해바라기 (sunflower)"},
            # 튤립 (Unsplash)
            "tulip_01.jpg": {"url": "https://images.unsplash.com/photo-1488928741225-2aaf732c96cc?w=640", "expected": "튤립 (tulip)"},
            "tulip_02.jpg": {"url": "https://images.unsplash.com/photo-1554494583-c4e1649bfe71?w=640", "expected": "튤립 (tulip)"},
            # 선인장 (Unsplash)
            "cactus_01.jpg": {"url": "https://images.unsplash.com/photo-1554631221-f9603e6808be?w=640", "expected": "선인장 (cactus)"},
            "cactus_02.jpg": {"url": "https://images.unsplash.com/photo-1517025423291-770fb99ae547?w=640", "expected": "선인장 (cactus)"},
            # 벚꽃 (Unsplash)
            "cherry_01.jpg": {"url": "https://images.unsplash.com/photo-1522383225653-ed111181a951?w=640", "expected": "벚꽃 (cherry blossom)"},
            "cherry_02.jpg": {"url": "https://images.unsplash.com/photo-1551829142-d9b8cf2c9232?w=640", "expected": "벚꽃 (cherry blossom)"},
            # 민들레 (Unsplash)
            "dandelion_01.jpg": {"url": "https://images.unsplash.com/photo-1544954412-78da2cfa1a0c?w=640", "expected": "민들레 (dandelion)"},
            "dandelion_02.jpg": {"url": "https://images.unsplash.com/photo-1533985062386-ef0837f31bc0?w=640", "expected": "민들레 (dandelion)"},
            # 라벤더 (Unsplash)
            "lavender_01.jpg": {"url": "https://images.unsplash.com/photo-1528756514091-dee5ecaa3278?w=640", "expected": "라벤더 (lavender)"},
        }
    },
}


# ============================================================================
# CLIP 분류기 프리셋 카테고리 (image_classifier.py에서 가져옴)
# ============================================================================
from image_classifier import PRESET_CATEGORIES


def download_images(category_data, output_dir):
    """카테고리별 이미지를 다운로드합니다."""
    downloaded = {}
    failed = []

    os.makedirs(output_dir, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    for filename, info in category_data["images"].items():
        filepath = os.path.join(output_dir, filename)

        # 이미 다운로드된 파일은 스킵
        if os.path.exists(filepath) and os.path.getsize(filepath) > 1000:
            downloaded[filename] = {"path": filepath, "expected": info["expected"]}
            continue

        try:
            response = requests.get(info["url"], headers=headers, timeout=15, stream=True)
            response.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # 이미지 유효성 검사
            try:
                img = Image.open(filepath)
                img.verify()
                downloaded[filename] = {"path": filepath, "expected": info["expected"]}
                print(f"  ✅ {filename} 다운로드 완료")
            except:
                os.remove(filepath)
                failed.append(filename)
                print(f"  ❌ {filename} 유효하지 않은 이미지")

        except Exception as e:
            failed.append(filename)
            print(f"  ❌ {filename} 다운로드 실패: {e}")

    return downloaded, failed


def run_accuracy_test(model, processor, device, category_name, category_data, downloaded_images):
    """특정 카테고리에 대한 정확도 테스트를 실행합니다."""
    preset = category_data["preset"]
    categories = PRESET_CATEGORIES[preset]["categories"]

    display_names = list(categories.keys())
    text_prompts = list(categories.values())

    correct = 0
    total = 0
    results_detail = []

    for filename, info in downloaded_images.items():
        filepath = info["path"]
        expected = info["expected"]

        try:
            img = Image.open(filepath).convert("RGB")
        except:
            print(f"  ⚠️ 이미지 로드 실패: {filename}")
            continue

        with torch.no_grad():
            inputs = processor(
                text=text_prompts,
                images=img,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]

        top_idx = probs.argmax().item()
        predicted = display_names[top_idx]
        confidence = probs[top_idx].item() * 100
        is_correct = predicted == expected

        if is_correct:
            correct += 1
        total += 1

        status = "✅" if is_correct else "❌"
        results_detail.append({
            "file": filename,
            "expected": expected,
            "predicted": predicted,
            "confidence": round(confidence, 2),
            "correct": is_correct,
        })
        print(f"  {status} {filename:<25s} | 정답: {expected:<20s} | 예측: {predicted:<20s} | 확신도: {confidence:.1f}%")

    accuracy = (correct / total * 100) if total > 0 else 0
    return accuracy, correct, total, results_detail


def main():
    print("=" * 70)
    print("🧪 CLIP 이미지 분류기 정확도 테스트")
    print("=" * 70)

    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] 디바이스: {device}")

    # CLIP 모델 로드
    model_name = "openai/clip-vit-base-patch32"
    print(f"[*] CLIP 모델 로드 중... ({model_name})")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print("[✓] 모델 로드 완료\n")

    # 테스트 이미지 디렉토리
    test_base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_images", "accuracy_test")
    os.makedirs(test_base_dir, exist_ok=True)

    all_results = {}
    overall_correct = 0
    overall_total = 0

    for category_name, category_data in TEST_IMAGES.items():
        print(f"\n{'='*70}")
        preset_info = PRESET_CATEGORIES[category_data["preset"]]
        print(f"📂 카테고리: {preset_info['name_ko']} ({category_name})")
        print(f"{'─'*70}")

        # 이미지 다운로드
        cat_dir = os.path.join(test_base_dir, category_name)
        print(f"\n[*] 이미지 다운로드 중...")
        downloaded, failed = download_images(category_data, cat_dir)

        if not downloaded:
            print(f"  ⚠️ 다운로드된 이미지가 없습니다. 스킵합니다.")
            continue

        print(f"\n[*] 다운로드 완료: {len(downloaded)}장 성공, {len(failed)}장 실패")

        # 정확도 테스트
        print(f"\n[*] 분류 테스트 중...")
        accuracy, correct, total, details = run_accuracy_test(
            model, processor, device, category_name, category_data, downloaded
        )

        overall_correct += correct
        overall_total += total

        all_results[category_name] = {
            "preset": category_data["preset"],
            "accuracy": round(accuracy, 2),
            "correct": correct,
            "total": total,
            "details": details,
        }

        print(f"\n{'─'*70}")
        print(f"📊 {preset_info['name_ko']} 정확도: {correct}/{total} ({accuracy:.1f}%)")

    # 전체 요약
    overall_accuracy = (overall_correct / overall_total * 100) if overall_total > 0 else 0

    print(f"\n\n{'='*70}")
    print(f"📋 전체 테스트 결과 요약")
    print(f"{'='*70}")
    print(f"{'카테고리':<20s} {'정확도':<15s} {'정답/전체':<15s}")
    print(f"{'─'*70}")

    for category_name, result in all_results.items():
        preset_info = PRESET_CATEGORIES[result["preset"]]
        name = preset_info["name_ko"]
        acc = f"{result['accuracy']:.1f}%"
        count = f"{result['correct']}/{result['total']}"
        print(f"{name:<20s} {acc:<15s} {count:<15s}")

    print(f"{'─'*70}")
    print(f"{'🏆 전체':<20s} {overall_accuracy:.1f}%{'':10s} {overall_correct}/{overall_total}")
    print(f"{'='*70}")

    # 오답 분석
    print(f"\n{'='*70}")
    print(f"📝 오답 분석")
    print(f"{'='*70}")
    has_wrong = False
    for category_name, result in all_results.items():
        wrong = [d for d in result["details"] if not d["correct"]]
        if wrong:
            has_wrong = True
            preset_info = PRESET_CATEGORIES[result["preset"]]
            print(f"\n  [{preset_info['name_ko']}]")
            for w in wrong:
                print(f"    ❌ {w['file']}: 정답={w['expected']}, 예측={w['predicted']} ({w['confidence']:.1f}%)")

    if not has_wrong:
        print("  ✅ 오답이 없습니다!")

    # 결과 JSON 저장
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "accuracy_test_results.json")
    summary = {
        "model": model_name,
        "device": str(device),
        "overall_accuracy": round(overall_accuracy, 2),
        "overall_correct": overall_correct,
        "overall_total": overall_total,
        "categories": all_results,
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] 결과가 저장되었습니다: {result_path}")


if __name__ == "__main__":
    main()
