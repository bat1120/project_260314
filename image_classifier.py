"""
이미지 분류 스크립트 (Image Classification Script) — CLIP Zero-shot & Auto-Pilot 버전
===================================================================================
OpenAI CLIP 모델을 사용하여 이미지를 커스텀 카테고리로 분류합니다.
Auto-Pilot 모드는 대분류(주제)를 먼저 파악한 후, 소분류(개별 사물)를 찾아내는 2단계 지능형 분류를 수행합니다.

사용법:
    # 🚀 Auto-Pilot 모드 (알아서 분류)
    python image_classifier.py --image 이미지.jpg
    python image_classifier.py --folder 이미지폴더

    # 프리셋 카테고리를 명시적으로 지정
    python image_classifier.py --image 이미지.jpg --preset animal

    # 커스텀 카테고리로 분류 지정
    python image_classifier.py --image 이미지.jpg --categories "고양이,강아지,새,물고기"

    # 기타 옵션
    python image_classifier.py --image 이미지.jpg --device cuda --top_k 3 --lang en
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

# Windows 환경 인코딩 이슈 대비
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


# ============================================================================
# 프리셋 카테고리 정의 (대분류를 위한 broad_prompt 포함)
# ============================================================================
PRESET_CATEGORIES = {
    "animal": {
        "name_ko": "🐾 동물",
        "name_en": "🐾 Animals",
        "broad_prompt": "a photo of an animal",
        "categories": {
            "개 (dog)": "a photo of a dog, a domestic pet canine",
            "고양이 (cat)": "a photo of a cute domestic house cat, a pet feline",
            "새 (bird)": "a photo of a bird, flying animal",
            "물고기 (fish)": "a photo of a fish, underwater animal",
            "말 (horse)": "a photo of a horse, equine",
            "곰 (bear)": "a photo of a bear, large wild animal",
            "사슴 (deer)": "a photo of a deer, wild animal with antlers",
            "토끼 (rabbit)": "a photo of a rabbit, bunny",
            "뱀 (snake)": "a photo of a snake, reptile",
            "거북이 (turtle)": "a photo of a turtle, reptile with a shell",
            "코끼리 (elephant)": "a photo of an elephant, large wild animal with trunk",
            "사자 (lion)": "a photo of a lion, wild feline predator",
            "호랑이 (tiger)": "a photo of a wild tiger, large wildlife predator with stripes",
            "원숭이 (monkey)": "a photo of a monkey, primate",
            "펭귄 (penguin)": "a photo of a penguin, flightless bird",
        },
    },
    "food": {
        "name_ko": "🍽️ 음식",
        "name_en": "🍽️ Food",
        "broad_prompt": "a photo of food",
        "categories": {
            "피자 (pizza)": "a photo of pizza",
            "스시 (sushi)": "a photo of sushi",
            "김치 (kimchi)": "a photo of kimchi",
            "비빔밥 (bibimbap)": "a photo of bibimbap",
            "햄버거 (hamburger)": "a photo of a hamburger",
            "파스타 (pasta)": "a photo of pasta",
            "샐러드 (salad)": "a photo of a salad",
            "스테이크 (steak)": "a photo of a steak",
            "라면 (ramen)": "a photo of ramen noodles",
            "케이크 (cake)": "a photo of a cake",
            "아이스크림 (ice cream)": "a photo of ice cream",
            "타코 (taco)": "a photo of a taco",
            "카레 (curry)": "a photo of curry",
            "떡볶이 (tteokbokki)": "a photo of tteokbokki",
            "치킨 (fried chicken)": "a photo of fried chicken",
        },
    },
    "plant": {
        "name_ko": "🌿 식물",
        "name_en": "🌿 Plants",
        "broad_prompt": "a photo of a plant or flower",
        "categories": {
            "장미 (rose)": "a photo of a rose flower",
            "해바라기 (sunflower)": "a photo of a sunflower",
            "벚꽃 (cherry blossom)": "a photo of cherry blossoms",
            "튤립 (tulip)": "a photo of tulips",
            "선인장 (cactus)": "a photo of a cactus",
            "소나무 (pine tree)": "a photo of a pine tree",
            "단풍나무 (maple)": "a photo of a maple tree",
            "대나무 (bamboo)": "a photo of bamboo",
            "연꽃 (lotus)": "a photo of a lotus flower",
            "라벤더 (lavender)": "a photo of lavender",
            "민들레 (dandelion)": "a photo of a dandelion",
            "난초 (orchid)": "a photo of an orchid",
            "백합 (lily)": "a photo of a lily flower",
            "무궁화 (hibiscus)": "a photo of a hibiscus flower",
            "코스모스 (cosmos)": "a photo of cosmos flowers",
        },
    },
    "vehicle": {
        "name_ko": "🚗 탈것",
        "name_en": "🚗 Vehicles",
        "broad_prompt": "a photo of a vehicle",
        "categories": {
            "자동차 (car)": "a photo of a car",
            "자전거 (bicycle)": "a photo of a bicycle",
            "비행기 (airplane)": "a photo of an airplane",
            "기차 (train)": "a photo of a train",
            "배 (ship)": "a photo of a ship",
            "오토바이 (motorcycle)": "a photo of a motorcycle",
            "버스 (bus)": "a photo of a bus",
            "트럭 (truck)": "a photo of a truck",
            "헬리콥터 (helicopter)": "a photo of a helicopter",
            "잠수함 (submarine)": "a photo of a submarine",
        },
    },
    "electronics": {
        "name_ko": "📱 전자제품",
        "name_en": "📱 Electronics",
        "broad_prompt": "a photo of an electronic device",
        "categories": {
            "스마트폰 (smartphone)": "a photo of a smartphone",
            "노트북 (laptop)": "a photo of a laptop computer",
            "키보드 (keyboard)": "a photo of a computer keyboard",
            "마우스 (mouse)": "a photo of a computer mouse",
            "모니터 (monitor)": "a photo of a computer monitor",
            "카메라 (camera)": "a photo of a camera",
            "헤드폰 (headphones)": "a photo of headphones",
            "태블릿 (tablet)": "a photo of a tablet device",
            "스피커 (speaker)": "a photo of a speaker",
            "게임 컨트롤러 (game controller)": "a photo of a game controller",
        },
    },
    "sports": {
        "name_ko": "⚽ 스포츠",
        "name_en": "⚽ Sports",
        "broad_prompt": "a photo about sports",
        "categories": {
            "축구 (soccer)": "a photo of people playing soccer",
            "농구 (basketball)": "a photo of people playing basketball",
            "야구 (baseball)": "a photo of people playing baseball",
            "테니스 (tennis)": "a photo of people playing tennis",
            "수영 (swimming)": "a photo of a person swimming",
            "스키 (skiing)": "a photo of a person skiing",
            "복싱 (boxing)": "a photo of people boxing",
            "요가 (yoga)": "a photo of a person doing yoga",
            "골프 (golf)": "a photo of a person playing golf",
            "배드민턴 (badminton)": "a photo of people playing badminton",
        },
    },
}


# ============================================================================
# 프리셋 목록 출력
# ============================================================================
def list_presets():
    """사용 가능한 프리셋 카테고리를 출력합니다."""
    print(f"\n{'='*60}")
    print(f"📋 사용 가능한 프리셋 카테고리")
    print(f"{'='*60}")
    for key, preset in PRESET_CATEGORIES.items():
        cats = list(preset["categories"].keys())
        cat_names = ", ".join(c.split(" (")[0] for c in cats[:5])
        print(f"\n  --preset {key:<15s} {preset['name_ko']}")
        print(f"       {cat_names} 등 {len(cats)}개 카테고리")
    print(f"\n{'='*60}")
    print(f"💡 커스텀 카테고리: --categories \"카테고리1,카테고리2,...\"")
    print(f"{'='*60}\n")


# ============================================================================
# CLIP 모델 로드
# ============================================================================
def load_clip_model(device):
    """
    CLIP 모델과 프로세서를 로드합니다.
    - openai/clip-vit-large-patch14-336 사용 (기존 224px보다 높은 336px 해상도로 미세한 특징을 더 잘 잡는 초고성능 모델)
    """
    model_name = "openai/clip-vit-large-patch14-336"
    print(f"[*] CLIP 모델 로드 중... ({model_name})")
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    print(f"[✓] CLIP 모델 로드 완료 (디바이스: {device})")
    return model, processor


def predict_category(model, processor, img, categories, device, top_k=5):
    """
    주어진 카테고리 딕셔너리에 대해 CLIP 예측을 수행합니다.
    🌟 업그레이드: Prompt Ensembling 기술 적용
    한 개의 단순한 문장 대신 여러 상황의 문장을 종합(평균)하여 정확도를 비약적으로 높입니다.
    """
    display_names = list(categories.keys())
    base_texts = list(categories.values())
    actual_top_k = min(top_k, len(display_names))

    # 앙상블을 위한 프롬프트 템플릿 (다양한 각도/상태의 객체를 포용)
    templates = [
        "{}",
        "a clean photo of a {}",
        "a close-up photo of a {}",
        "a bright photo of a {}",
        "a cropped photo of a {}",
        "a good photo of a {}",
        "a beautiful photo of a {}",
        "a photo of a large {}",
        "a photo of a small {}",
        "a clear photo of a {}"
    ]

    start_time = time.time()
    with torch.no_grad():
        # 이미지 피처 추출
        image_inputs = processor(images=img, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # 텍스트 피처 추출 (앙상블)
        text_features_list = []
        for text in base_texts:
            # 기본 텍스트에 템플릿 적용 (예: "a clean photo of a photo of a dog"가 될 수도 있지만 특징은 강화됨)
            # 깔끔함을 위해 원래 텍스트가 "a photo of"로 시작하면 교체하거나 그대로 앙상블
            clean_text = text.replace("a photo of ", "")
            ensemble_texts = [template.format(clean_text) for template in templates]
            
            text_inputs = processor(text=ensemble_texts, return_tensors="pt", padding=True).to(device)
            text_embeddings = model.get_text_features(**text_inputs)
            
            # 템플릿 결과들의 평균을 내어 정규화
            text_features_mean = text_embeddings.mean(dim=0, keepdim=True)
            text_features_mean = text_features_mean / text_features_mean.norm(dim=-1, keepdim=True)
            text_features_list.append(text_features_mean)

        text_features = torch.cat(text_features_list, dim=0) # [num_categories, feature_dim]
        
        # 코사인 유사도에 스케일 곱하기 (logit scale)
        logit_scale = model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        
        probs = logits_per_image.softmax(dim=1)[0]
        
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000

    top_probs, top_indices = torch.topk(probs, actual_top_k)
    return top_probs, top_indices, display_names, inference_time


def format_name(name, lang="ko"):
    if lang == "en":
        # "개 (dog)" → "dog"
        if "(" in name and ")" in name:
            return name.split("(")[1].rstrip(")")
        return name
    return name


# ============================================================================
# 단일 이미지 분류 (지정 모드: --preset 또는 --categories 사용 시)
# ============================================================================
def classify_image(model, processor, image_path, categories, device, top_k=5, lang="ko"):
    """사용자가 지정한 카테고리(프리셋 또는 커스텀) 1단계로만 분류합니다."""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[✗] 이미지 로드 실패: {image_path} - {e}")
        return None

    top_probs, top_indices, display_names, inference_time = predict_category(
        model, processor, img, categories, device, top_k
    )

    actual_top_k = len(top_probs)
    results = {
        "file": os.path.basename(image_path),
        "path": str(image_path),
        "mode": "manual",
        "top_prediction": {
            "class": format_name(display_names[top_indices[0].item()], lang),
            "confidence": round(top_probs[0].item() * 100, 2),
        },
        "top_k": [],
        "inference_time_ms": round(inference_time, 2),
    }

    for i in range(actual_top_k):
        idx = top_indices[i].item()
        prob = top_probs[i].item() * 100
        results["top_k"].append({
            "rank": i + 1,
            "class": format_name(display_names[idx], lang),
            "confidence": round(prob, 2),
        })

    return results


# ============================================================================
# 단일 이미지 분류 (Auto-Pilot 모드: 2단계 지능형 분류)
# ============================================================================
def classify_image_auto(model, processor, image_path, device, top_k=5, lang="ko"):
    """
    1단계: [동물, 음식, 식물, 탈것 등] 대분류 탐색
    2단계: 선택된 대분류의 하위 카테고리 중 세부 사물 예측
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[✗] 이미지 로드 실패: {image_path} - {e}")
        return None

    total_inference_time = 0

    # 1. 대분류 식별 (Stage 1)
    broad_categories = {key: preset["broad_prompt"] for key, preset in PRESET_CATEGORIES.items()}
    bb_probs, bb_indices, bb_display_names, t1 = predict_category(
        model, processor, img, broad_categories, device, top_k=1
    )
    total_inference_time += t1

    best_preset_key = bb_display_names[bb_indices[0].item()]
    broad_confidence = bb_probs[0].item() * 100

    # 2. 소분류 식별 (Stage 2)
    fine_categories = PRESET_CATEGORIES[best_preset_key]["categories"]
    top_probs, top_indices, display_names, t2 = predict_category(
        model, processor, img, fine_categories, device, top_k
    )
    total_inference_time += t2

    actual_top_k = len(top_probs)
    broad_name = PRESET_CATEGORIES[best_preset_key]["name_ko" if lang == "ko" else "name_en"]

    results = {
        "file": os.path.basename(image_path),
        "path": str(image_path),
        "mode": "auto",
        "broad_prediction": {
            "key": best_preset_key,
            "class": broad_name,
            "confidence": round(broad_confidence, 2),
        },
        "top_prediction": {
            "class": format_name(display_names[top_indices[0].item()], lang),
            "confidence": round(top_probs[0].item() * 100, 2),
        },
        "top_k": [],
        "inference_time_ms": round(total_inference_time, 2),
    }

    for i in range(actual_top_k):
        idx = top_indices[i].item()
        prob = top_probs[i].item() * 100
        results["top_k"].append({
            "rank": i + 1,
            "class": format_name(display_names[idx], lang),
            "confidence": round(prob, 2),
        })

    return results


# ============================================================================
# 결과 출력
# ============================================================================
def print_results(results):
    """분류 결과를 보기 좋게 출력합니다. (Auto-Pilot 지원)"""
    if results is None:
        return

    print(f"\n{'='*65}")
    print(f"📁 파일: {results['file']}")
    print(f"{'─'*65}")
    
    if results.get("mode") == "auto":
        print(f"🧩 대분류: {results['broad_prediction']['class']} (확신도: {results['broad_prediction']['confidence']:.2f}%)")
        print(f"🎯 소분류: {results['top_prediction']['class']} (확신도: {results['top_prediction']['confidence']:.2f}%)")
    else:
        print(f"🏆 예측 결과: {results['top_prediction']['class']}")
        print(f"   확신도: {results['top_prediction']['confidence']:.2f}%")
        
    print(f"⏱️  추론 시간: {results['inference_time_ms']:.2f} ms")
    print(f"{'─'*65}")
    print(f"📊 Top-{len(results['top_k'])} 상세 랭킹:")
    for item in results["top_k"]:
        bar_len = int(item["confidence"] / 2)  # 최대 50칸
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"   {item['rank']:2d}. {item['class']:<30s} {item['confidence']:6.2f}% |{bar}|")
    print(f"{'='*65}")


# ============================================================================
# 폴더 내 이미지 일괄 분류
# ============================================================================
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

def classify_folder(model, processor, folder_path, device, top_k=5, lang="ko", preset=None, custom_categories=None):
    """폴더 내 모든 이미지를 분류합니다."""
    folder = Path(folder_path)
    if not folder.is_dir():
        print(f"[✗] 폴더를 찾을 수 없습니다: {folder_path}")
        return []

    image_files = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"[✗] 폴더에 이미지가 없습니다: {folder_path}")
        print(f"    지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}")
        return []

    print(f"\n[*] 총 {len(image_files)}개의 이미지를 분류합니다...")
    
    # 카테고리 수동 지정 여부 확인
    manual_categories = None
    if custom_categories:
        cats = [c.strip() for c in custom_categories.split(",") if c.strip()]
        manual_categories = {cat: f"a photo of {cat}" for cat in cats}
    elif preset:
        manual_categories = PRESET_CATEGORIES[preset]["categories"]

    all_results = []
    total_time = 0

    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 처리 중: {img_path.name}")
        
        if manual_categories:
            result = classify_image(model, processor, str(img_path), manual_categories, device, top_k, lang)
        else:
            result = classify_image_auto(model, processor, str(img_path), device, top_k, lang)
            
        if result:
            all_results.append(result)
            print_results(result)
            total_time += result["inference_time_ms"]

    # 요약
    if all_results:
        print(f"\n{'='*65}")
        print(f"📋 분류 요약")
        print(f"{'─'*65}")
        print(f"  총 이미지 수: {len(all_results)}개")
        print(f"  총 추론 시간: {total_time:.2f} ms")
        print(f"  평균 추론 시간: {total_time / len(all_results):.2f} ms / 이미지")

        # 카테고리 분포
        category_counts = {}
        for r in all_results:
            cls = r["top_prediction"]["class"]
            if "broad_prediction" in r:
                cls = f"[{r['broad_prediction']['class']}] {cls}"
            category_counts[cls] = category_counts.get(cls, 0) + 1

        print(f"\n  📊 카테고리 분포:")
        for cls, count in sorted(category_counts.items(), key=lambda x: -x[1]):
            pct = count / len(all_results) * 100
            print(f"     {cls:<35s}: {count:3d}개 ({pct:.1f}%)")

        print(f"{'='*65}")

    return all_results


# ============================================================================
# 결과 저장 (JSON)
# ============================================================================
def save_results_to_json(results, output_path):
    """분류 결과를 JSON 파일로 저장합니다."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[✓] 결과가 저장되었습니다: {output_path}")


# ============================================================================
# 메인 함수
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="CLIP 기반 Zero-shot 스마트 이미지 분류기 (Auto-Pilot 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 🚀 (추천) 알아서 분류 - Auto-Pilot 모드
  python image_classifier.py --image photo.jpg

  # 프리셋 카테고리로 강제 지정
  python image_classifier.py --image cat.jpg --preset animal

  # 커스텀 카테고리 사용
  python image_classifier.py --image img.jpg --categories "고양이,강아지,새"

  # 폴더 전체 자동 분류
  python image_classifier.py --folder ./images

  # 프리셋 목록 확인
  python image_classifier.py --list-presets
        """,
    )
    parser.add_argument("--image", type=str, help="분류할 이미지 파일 경로")
    parser.add_argument("--folder", type=str, help="분류할 이미지가 있는 폴더 경로")
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESET_CATEGORIES.keys()),
        help=f"프리셋 카테고리를 강제 지정할 때 사용",
    )
    parser.add_argument(
        "--categories",
        type=str,
        help='커스텀 카테고리를 강제 지정할 때 사용 (쉼표 구분, 예: "고양이,강아지,새")',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="사용할 디바이스 (기본값: auto)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="상위 K개 예측 결과 표시 (기본값: 5)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="ko",
        choices=["ko", "en"],
        help="출력 언어 (기본값: ko - 한국어)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="결과를 저장할 JSON 파일 경로 (선택)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="사용 가능한 프리셋 카테고리 목록 출력",
    )

    args = parser.parse_args()

    # 프리셋 목록 출력
    if args.list_presets:
        list_presets()
        sys.exit(0)

    # 입력 검증
    if not args.image and not args.folder:
        parser.print_help()
        print("\n[✗] --image 또는 --folder 인자를 지정해주세요.")
        sys.exit(1)

    if args.image and args.folder:
        print("[✗] --image와 --folder 중 하나만 지정해주세요.")
        sys.exit(1)

    # 디바이스 설정
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n[*] 디바이스: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")

    auto_mode = not args.preset and not args.categories
    if auto_mode:
        print(f"\n🚀 [Auto-Pilot 모드] 대분류 및 소분류를 자동으로 탐색합니다.")
    else:
        print(f"\n⚙️ [수동 모드] 지정된 카테고리로만 분류를 수행합니다.")
        if args.preset:
            print(f"    선택된 프리셋: {PRESET_CATEGORIES[args.preset]['name_ko']}")
        elif args.categories:
            print(f"    선택된 커스텀 카테고리: {args.categories}")

    # 모델 로드
    model, processor = load_clip_model(device)

    # 분류 실행
    if args.image:
        if not os.path.isfile(args.image):
            print(f"[✗] 파일을 찾을 수 없습니다: {args.image}")
            sys.exit(1)
            
        if auto_mode:
            result = classify_image_auto(model, processor, args.image, device, args.top_k, args.lang)
        else:
            cats = None
            if args.preset:
                cats = PRESET_CATEGORIES[args.preset]["categories"]
            else:
                c_list = [c.strip() for c in args.categories.split(",") if c.strip()]
                cats = {cat: f"a photo of {cat}" for cat in c_list}
            result = classify_image(model, processor, args.image, cats, device, args.top_k, args.lang)
            
        print_results(result)
        all_results = [result] if result else []
    else:
        all_results = classify_folder(model, processor, args.folder, device, args.top_k, args.lang, args.preset, args.categories)

    # 결과 저장
    if args.save and all_results:
        save_results_to_json(all_results, args.save)


if __name__ == "__main__":
    main()
