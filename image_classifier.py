"""
이미지 분류 스크립트 (Image Classification Script)
=================================================
MobileNet V3 Small 모델을 사용하여 이미지를 분류합니다.
ImageNet 1000개 클래스 기반으로 예측합니다.

사용법:
    # 단일 이미지 분류
    python image_classifier.py --image 이미지경로.jpg

    # 폴더 내 전체 이미지 분류
    python image_classifier.py --folder 이미지폴더경로

    # GPU 사용 (가능한 경우)
    python image_classifier.py --image 이미지경로.jpg --device cuda

    # Top-K 결과 표시 (기본값: 5)
    python image_classifier.py --image 이미지경로.jpg --top_k 10
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


# ============================================================================
# ImageNet 클래스 레이블 로드
# ============================================================================
def load_imagenet_labels():
    """ImageNet 1000개 클래스 레이블을 로드합니다."""
    # torchvision의 MobileNet V3 Small weights에서 meta 정보 가져오기
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    categories = weights.meta["categories"]
    return categories


# ============================================================================
# 이미지 전처리
# ============================================================================
def get_preprocess_transform():
    """MobileNet V3 Small에 맞는 이미지 전처리 파이프라인을 반환합니다."""
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()
    return preprocess


# ============================================================================
# 모델 로드
# ============================================================================
def load_model(device):
    """
    MobileNet V3 Small 모델을 로드합니다.
    - 사전 학습된 ImageNet 가중치를 사용합니다.
    - 평가(eval) 모드로 설정합니다.
    """
    print("[*] MobileNet V3 Small 모델 로드 중...")
    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = models.mobilenet_v3_small(weights=weights)
    model = model.to(device)
    model.eval()
    print(f"[✓] 모델 로드 완료 (디바이스: {device})")
    return model


# ============================================================================
# 단일 이미지 분류
# ============================================================================
def classify_image(model, image_path, preprocess, device, labels, top_k=5):
    """
    단일 이미지를 분류합니다.

    Args:
        model: 분류 모델
        image_path: 이미지 파일 경로
        preprocess: 전처리 변환
        device: 디바이스 (cpu/cuda)
        labels: 클래스 레이블 리스트
        top_k: 상위 K개 결과 반환

    Returns:
        dict: 분류 결과 (파일명, 예측 클래스, 확률, Top-K 결과, 추론 시간)
    """
    # 이미지 로드 및 전처리
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"[✗] 이미지 로드 실패: {image_path} - {e}")
        return None

    input_tensor = preprocess(img).unsqueeze(0).to(device)  # 배치 차원 추가

    # 추론
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # ms 단위

    # Softmax로 확률 변환
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Top-K 결과
    top_probs, top_indices = torch.topk(probabilities, top_k)

    results = {
        "file": os.path.basename(image_path),
        "path": str(image_path),
        "top_prediction": {
            "class": labels[top_indices[0].item()],
            "confidence": top_probs[0].item() * 100,
        },
        "top_k": [],
        "inference_time_ms": round(inference_time, 2),
    }

    for i in range(top_k):
        idx = top_indices[i].item()
        prob = top_probs[i].item() * 100
        results["top_k"].append({
            "rank": i + 1,
            "class": labels[idx],
            "confidence": round(prob, 2),
        })

    return results


# ============================================================================
# 결과 출력
# ============================================================================
def print_results(results):
    """분류 결과를 보기 좋게 출력합니다."""
    if results is None:
        return

    print(f"\n{'='*60}")
    print(f"📁 파일: {results['file']}")
    print(f"{'─'*60}")
    print(f"🏆 예측 결과: {results['top_prediction']['class']}")
    print(f"   확신도: {results['top_prediction']['confidence']:.2f}%")
    print(f"⏱️  추론 시간: {results['inference_time_ms']:.2f} ms")
    print(f"{'─'*60}")
    print(f"📊 Top-{len(results['top_k'])} 예측:")
    for item in results["top_k"]:
        bar_len = int(item["confidence"] / 2)  # 최대 50칸
        bar = "█" * bar_len + "░" * (50 - bar_len)
        print(f"   {item['rank']:2d}. {item['class']:<30s} {item['confidence']:6.2f}% |{bar}|")
    print(f"{'='*60}")


# ============================================================================
# 폴더 내 이미지 일괄 분류
# ============================================================================
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def classify_folder(model, folder_path, preprocess, device, labels, top_k=5):
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
    print(f"    지원 형식: {', '.join(SUPPORTED_EXTENSIONS)}")

    all_results = []
    total_time = 0

    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 처리 중: {img_path.name}")
        result = classify_image(model, str(img_path), preprocess, device, labels, top_k)
        if result:
            all_results.append(result)
            print_results(result)
            total_time += result["inference_time_ms"]

    # 요약
    if all_results:
        print(f"\n{'='*60}")
        print(f"📋 분류 요약")
        print(f"{'─'*60}")
        print(f"  총 이미지 수: {len(all_results)}개")
        print(f"  총 추론 시간: {total_time:.2f} ms")
        print(f"  평균 추론 시간: {total_time / len(all_results):.2f} ms / 이미지")
        print(f"{'='*60}")

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
        description="MobileNet V3 Small 기반 이미지 분류기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python image_classifier.py --image cat.jpg
  python image_classifier.py --folder ./images
  python image_classifier.py --image dog.png --device cuda --top_k 10
  python image_classifier.py --folder ./images --save results.json
        """,
    )
    parser.add_argument("--image", type=str, help="분류할 이미지 파일 경로")
    parser.add_argument("--folder", type=str, help="분류할 이미지가 있는 폴더 경로")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="사용할 디바이스 (기본값: auto - GPU 가능하면 GPU 사용)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="상위 K개 예측 결과 표시 (기본값: 5)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="결과를 저장할 JSON 파일 경로 (선택)",
    )

    args = parser.parse_args()

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

    print(f"[*] 디바이스: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 모델 및 전처리 로드
    model = load_model(device)
    preprocess = get_preprocess_transform()
    labels = load_imagenet_labels()

    # GPU 워밍업 (정확한 시간 측정을 위해)
    if device.type == "cuda":
        print("[*] GPU 워밍업 중...")
        dummy = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy)
        print("[✓] GPU 워밍업 완료")

    # 분류 실행
    if args.image:
        if not os.path.isfile(args.image):
            print(f"[✗] 파일을 찾을 수 없습니다: {args.image}")
            sys.exit(1)
        result = classify_image(model, args.image, preprocess, device, labels, args.top_k)
        print_results(result)
        all_results = [result] if result else []
    else:
        all_results = classify_folder(model, args.folder, preprocess, device, labels, args.top_k)

    # 결과 저장
    if args.save and all_results:
        save_results_to_json(all_results, args.save)


if __name__ == "__main__":
    main()
