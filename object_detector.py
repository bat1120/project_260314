"""
Object Detection 스크립트 (Object Detection Script)
===================================================
Faster R-CNN (ResNet50 FPN v2) 모델을 사용하여
이미지에서 객체를 탐지하고 시각화합니다.
COCO 데이터셋 91개 클래스 기반으로 예측합니다.

사용법:
    # 단일 이미지 객체 탐지
    python object_detector.py --image 이미지경로.jpg

    # 폴더 내 전체 이미지 객체 탐지
    python object_detector.py --folder 이미지폴더경로

    # 신뢰도 임계값 설정 (기본값: 0.5)
    python object_detector.py --image 이미지경로.jpg --threshold 0.3

    # GPU 사용 (가능한 경우)
    python object_detector.py --image 이미지경로.jpg --device cuda

    # 시각화 결과만 저장 (팝업 표시 안함)
    python object_detector.py --image 이미지경로.jpg --no-show
"""

import argparse
import os
import sys
import time
import json
from pathlib import Path

import torch
import torchvision.models.detection as detection_models
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# ============================================================================
# COCO 클래스 레이블 (91 카테고리)
# ============================================================================
COCO_LABELS = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A",
    "N/A", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "N/A", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "N/A", "dining table", "N/A", "N/A", "toilet", "N/A",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ============================================================================
# 시각화용 색상 팔레트 (클래스별 고유 색상)
# ============================================================================
def generate_color_palette(num_colors=91):
    """클래스별로 구분되는 시각적으로 뚜렷한 색상 팔레트를 생성합니다."""
    colors = []
    for i in range(num_colors):
        hue = i / num_colors
        # HSV -> RGB 변환 (채도와 밝기를 높게)
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        colors.append((r, g, b))
    return colors


COLOR_PALETTE = generate_color_palette()


# ============================================================================
# 모델 로드
# ============================================================================
# 모델 설명
MODEL_INFO = {
    "fasterrcnn_resnet50": {
        "name": "Faster R-CNN ResNet50 FPN v2",
        "desc": "고정확도 모델 (COCO mAP: 46.7)",
    },
    "fasterrcnn_mobilenet": {
        "name": "Faster R-CNN MobileNet V3 Large 320 FPN",
        "desc": "경량 모델 - 빠르지만 정확도 낮음 (COCO mAP: 22.8)",
    },
    "ssdlite": {
        "name": "SSDLite320 MobileNet V3 Large",
        "desc": "초경량 모델 - 가장 빠름 (COCO mAP: 21.3)",
    },
}


def load_model(device, model_name="fasterrcnn_resnet50"):
    """
    Object Detection 모델을 로드합니다.

    Args:
        device: 디바이스 (cpu/cuda)
        model_name: 모델 이름
            - 'fasterrcnn_resnet50': 고정확도 (기본값, 권장)
            - 'fasterrcnn_mobilenet': 경량/빠름
            - 'ssdlite': 초경량/가장 빠름

    Returns:
        로드된 모델
    """
    info = MODEL_INFO.get(model_name, {})
    print(f"[*] Object Detection 모델 로드 중...")
    print(f"    모델: {info.get('name', model_name)}")
    print(f"    설명: {info.get('desc', '')}")

    if model_name == "ssdlite":
        weights = detection_models.SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        model = detection_models.ssdlite320_mobilenet_v3_large(weights=weights)
    elif model_name == "fasterrcnn_mobilenet":
        weights = detection_models.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        model = detection_models.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    else:  # fasterrcnn_resnet50 (기본 - 고정확도)
        weights = detection_models.FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        model = detection_models.fasterrcnn_resnet50_fpn_v2(weights=weights)

    model = model.to(device)
    model.eval()
    print(f"[✓] 모델 로드 완료 (디바이스: {device})")
    return model


# ============================================================================
# 이미지 전처리
# ============================================================================
def preprocess_image(image_path, device):
    """이미지를 로드하고 모델 입력 형태로 변환합니다."""
    img = Image.open(image_path).convert("RGB")
    img_tensor = F.to_tensor(img).to(device)
    return img, img_tensor


# ============================================================================
# 객체 탐지 실행
# ============================================================================
def detect_objects(model, img_tensor, threshold=0.5):
    """
    이미지에서 객체를 탐지합니다.

    Args:
        model: Detection 모델
        img_tensor: 전처리된 이미지 텐서
        threshold: 신뢰도 임계값

    Returns:
        dict: 탐지 결과 (boxes, labels, scores)
    """
    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]

    # 신뢰도 임계값 이상인 결과만 필터링
    mask = pred["scores"] >= threshold
    filtered = {
        "boxes": pred["boxes"][mask].cpu().numpy(),
        "labels": pred["labels"][mask].cpu().numpy(),
        "scores": pred["scores"][mask].cpu().numpy(),
    }

    return filtered


# ============================================================================
# 시각화 (matplotlib)
# ============================================================================
def visualize_detections(image, detections, output_path, show=True):
    """
    탐지 결과를 이미지 위에 시각화합니다.

    Args:
        image: PIL Image
        detections: 탐지 결과 dict
        output_path: 결과 이미지 저장 경로
        show: 화면에 표시할지 여부
    """
    fig, ax = plt.subplots(1, figsize=(16, 12))
    ax.imshow(image)

    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]

    num_detections = len(boxes)

    for i in range(num_detections):
        box = boxes[i]
        label_idx = labels[i]
        score = scores[i]

        class_name = COCO_LABELS[label_idx] if label_idx < len(COCO_LABELS) else f"class_{label_idx}"
        color = COLOR_PALETTE[label_idx % len(COLOR_PALETTE)]

        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1

        # 바운딩 박스 그리기
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2.5,
            edgecolor=color,
            facecolor="none",
            linestyle="-",
        )
        ax.add_patch(rect)

        # 레이블 배경 박스
        label_text = f"{class_name} {score * 100:.1f}%"
        text_size = max(10, min(14, int(image.size[0] / 60)))

        # 텍스트 배경
        ax.text(
            x1, y1 - 4,
            label_text,
            fontsize=text_size,
            fontweight="bold",
            color="white",
            verticalalignment="bottom",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=color,
                edgecolor=color,
                alpha=0.85,
            ),
        )

    # 제목
    ax.set_title(
        f"Object Detection Result  |  {num_detections} objects detected",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.1)
    print(f"[✓] 시각화 저장: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ============================================================================
# PIL 기반 시각화 (폰트 없이도 동작)
# ============================================================================
def visualize_detections_pil(image, detections, output_path):
    """
    PIL를 사용하여 탐지 결과를 시각화합니다. (matplotlib 없이도 동작)

    Args:
        image: PIL Image
        detections: 탐지 결과 dict
        output_path: 결과 이미지 저장 경로
    """
    import colorsys

    draw_img = image.copy()
    draw = ImageDraw.Draw(draw_img)

    # 폰트 로드 시도
    try:
        font_size = max(14, min(24, image.size[0] // 40))
        font = ImageFont.truetype("arial.ttf", font_size)
        small_font = ImageFont.truetype("arial.ttf", font_size - 4)
    except (IOError, OSError):
        font = ImageFont.load_default()
        small_font = font

    boxes = detections["boxes"]
    labels = detections["labels"]
    scores = detections["scores"]

    for i in range(len(boxes)):
        box = boxes[i]
        label_idx = labels[i]
        score = scores[i]

        class_name = COCO_LABELS[label_idx] if label_idx < len(COCO_LABELS) else f"class_{label_idx}"

        # 클래스별 색상
        hue = label_idx / len(COCO_LABELS)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        color = (int(r * 255), int(g * 255), int(b * 255))

        x1, y1, x2, y2 = [int(v) for v in box]

        # 바운딩 박스 (두께 3)
        for offset in range(3):
            draw.rectangle(
                [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                outline=color,
            )

        # 레이블 텍스트
        label_text = f"{class_name} {score * 100:.1f}%"
        text_bbox = draw.textbbox((0, 0), label_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        # 레이블 배경
        draw.rectangle(
            [x1, y1 - text_h - 8, x1 + text_w + 8, y1],
            fill=color,
        )
        draw.text((x1 + 4, y1 - text_h - 6), label_text, fill="white", font=font)

    draw_img.save(output_path, quality=95)
    print(f"[✓] 시각화 저장 (PIL): {output_path}")
    return draw_img


# ============================================================================
# 단일 이미지 처리
# ============================================================================
def process_image(model, image_path, device, threshold, output_dir, show=True, model_name="fasterrcnn"):
    """단일 이미지에 대해 객체 탐지를 수행하고 시각화합니다."""

    if not os.path.isfile(image_path):
        print(f"[✗] 파일을 찾을 수 없습니다: {image_path}")
        return None

    print(f"\n{'='*60}")
    print(f"📁 파일: {os.path.basename(image_path)}")
    print(f"{'─'*60}")

    # 이미지 로드
    img, img_tensor = preprocess_image(image_path, device)
    print(f"   이미지 크기: {img.size[0]} x {img.size[1]}")

    # 객체 탐지
    start_time = time.time()
    detections = detect_objects(model, img_tensor, threshold)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000

    num_objects = len(detections["boxes"])
    print(f"   탐지된 객체: {num_objects}개")
    print(f"   추론 시간: {inference_time:.2f} ms")
    print(f"   신뢰도 임계값: {threshold}")

    # 탐지 결과 출력
    if num_objects > 0:
        print(f"\n   {'순위':<6}{'클래스':<20}{'신뢰도':<10}{'위치 (x1,y1,x2,y2)'}")
        print(f"   {'─'*56}")
        for i in range(num_objects):
            label_idx = detections["labels"][i]
            class_name = COCO_LABELS[label_idx] if label_idx < len(COCO_LABELS) else f"class_{label_idx}"
            score = detections["scores"][i] * 100
            box = detections["boxes"][i]
            print(f"   {i+1:<6}{class_name:<20}{score:<10.1f}({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})")
    else:
        print("\n   ⚠️  탐지된 객체가 없습니다. 임계값을 낮춰보세요 (--threshold 0.3)")

    # 시각화 저장
    os.makedirs(output_dir, exist_ok=True)
    basename = Path(image_path).stem
    output_path_mpl = os.path.join(output_dir, f"{basename}_detected.png")
    output_path_pil = os.path.join(output_dir, f"{basename}_detected_pil.png")

    # matplotlib 시각화
    try:
        visualize_detections(img, detections, output_path_mpl, show=show)
    except Exception as e:
        print(f"   [!] matplotlib 시각화 실패: {e}")

    # PIL 시각화 (항상 생성)
    visualize_detections_pil(img, detections, output_path_pil)

    print(f"{'='*60}")

    # 결과 반환
    result = {
        "file": os.path.basename(image_path),
        "path": str(image_path),
        "image_size": {"width": img.size[0], "height": img.size[1]},
        "num_objects": num_objects,
        "inference_time_ms": round(inference_time, 2),
        "threshold": threshold,
        "model": model_name,
        "detections": [],
    }

    for i in range(num_objects):
        label_idx = int(detections["labels"][i])
        result["detections"].append({
            "class": COCO_LABELS[label_idx] if label_idx < len(COCO_LABELS) else f"class_{label_idx}",
            "confidence": round(float(detections["scores"][i]) * 100, 2),
            "bbox": {
                "x1": round(float(detections["boxes"][i][0]), 1),
                "y1": round(float(detections["boxes"][i][1]), 1),
                "x2": round(float(detections["boxes"][i][2]), 1),
                "y2": round(float(detections["boxes"][i][3]), 1),
            },
        })

    return result


# ============================================================================
# 메인 함수
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Faster R-CNN / SSDLite 기반 Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python object_detector.py --image dog.jpg
  python object_detector.py --folder ./images --threshold 0.3
  python object_detector.py --image street.png --model ssdlite --device cuda
  python object_detector.py --folder ./images --save detection_results.json --no-show
        """,
    )
    parser.add_argument("--image", type=str, help="탐지할 이미지 파일 경로")
    parser.add_argument("--folder", type=str, help="탐지할 이미지가 있는 폴더 경로")
    parser.add_argument(
        "--model",
        type=str,
        default="fasterrcnn_resnet50",
        choices=["fasterrcnn_resnet50", "fasterrcnn_mobilenet", "ssdlite"],
        help="사용할 모델 (기본값: fasterrcnn_resnet50 - 고정확도)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="사용할 디바이스 (기본값: auto)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="객체 탐지 신뢰도 임계값 (기본값: 0.5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="detection_results",
        help="결과 이미지 저장 폴더 (기본값: detection_results)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="결과를 저장할 JSON 파일 경로 (선택)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="시각화 팝업을 표시하지 않음 (저장만)",
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

    # matplotlib 백엔드 설정
    if args.no_show:
        matplotlib.use("Agg")

    # 디바이스 설정
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"[*] 디바이스: {device}")
    if device.type == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 모델 로드
    model = load_model(device, args.model)

    # GPU 워밍업
    if device.type == "cuda":
        print("[*] GPU 워밍업 중...")
        warmup_size = 320 if args.model in ["ssdlite", "fasterrcnn_mobilenet"] else 640
        dummy = torch.randn(3, warmup_size, warmup_size).to(device)
        with torch.no_grad():
            for _ in range(3):
                _ = model([dummy])
        print("[✓] GPU 워밍업 완료")

    # 탐지 실행
    show = not args.no_show
    all_results = []

    if args.image:
        result = process_image(
            model, args.image, device, args.threshold,
            args.output, show=show, model_name=args.model,
        )
        if result:
            all_results.append(result)

    else:  # --folder
        folder = Path(args.folder)
        if not folder.is_dir():
            print(f"[✗] 폴더를 찾을 수 없습니다: {args.folder}")
            sys.exit(1)

        supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        image_files = [
            f for f in sorted(folder.iterdir())
            if f.is_file() and f.suffix.lower() in supported
        ]

        if not image_files:
            print(f"[✗] 폴더에 이미지가 없습니다: {args.folder}")
            sys.exit(1)

        print(f"\n[*] 총 {len(image_files)}개의 이미지를 처리합니다...")

        total_time = 0
        total_objects = 0

        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] 처리 중...")
            result = process_image(
                model, str(img_path), device, args.threshold,
                args.output, show=show, model_name=args.model,
            )
            if result:
                all_results.append(result)
                total_time += result["inference_time_ms"]
                total_objects += result["num_objects"]

        # 요약
        if all_results:
            print(f"\n{'='*60}")
            print(f"📋 Object Detection 요약")
            print(f"{'─'*60}")
            print(f"  모델: {args.model}")
            print(f"  총 이미지 수: {len(all_results)}개")
            print(f"  총 탐지 객체: {total_objects}개")
            print(f"  총 추론 시간: {total_time:.2f} ms")
            print(f"  평균 추론 시간: {total_time / len(all_results):.2f} ms / 이미지")
            print(f"{'='*60}")

    # JSON 저장
    if args.save and all_results:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n[✓] 결과가 저장되었습니다: {args.save}")


if __name__ == "__main__":
    main()
