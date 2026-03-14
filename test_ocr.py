"""
한글/영문 혼합 문서 OCR 인식 + 문맥 기반 번역 + 가독성 시각화 스크립트

- OCR 엔진: EasyOCR (ko, en)
- 번역 엔진: translators (Bing) — 문맥에 맞는 자연스러운 번역 제공
- 시각화: Pillow 기반, 원본 텍스트를 덮어쓰고 번역된 한글을 깔끔하게 배치
"""

import os
import sys
import argparse
import requests
import time
import re

import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# stdout 인코딩을 UTF-8로 설정 (윈도우 환경 대응)
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ─── 설정 ─────────────────────────────────────
DEFAULT_IMAGE_URL = 'https://raw.githubusercontent.com/JaidedAI/EasyOCR/master/examples/korean.png'
DEFAULT_IMAGE_PATH = 'korean_sample.png'
OUTPUT_VISUAL = 'ocr_translated_visualized.jpg'
OUTPUT_TEXT   = 'ocr_translated_result.txt'
# ───────────────────────────────────────────────


def download_image(url, save_path):
    """URL에서 이미지를 다운로드"""
    if not os.path.exists(save_path):
        print(f"[다운로드] {url}")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(resp.content)
        print("[다운로드] 완료!")
    else:
        print("[다운로드] 이미 존재하는 이미지를 사용합니다.")


def is_english(text):
    """텍스트에 영문 알파벳이 주된 구성인지 판단"""
    alpha = re.findall(r'[a-zA-Z]', text)
    hangul = re.findall(r'[가-힣]', text)
    # 영문 글자가 한글보다 많을 때만 번역 대상
    return len(alpha) > len(hangul) and len(alpha) >= 2


def contextual_translate(texts_to_translate):
    """
    문맥 기반 번역: 개별 문장을 번역하되, 단일 단어나 짧은 구문은
    문서 맥락 힌트를 붙여서 번역 품질을 높입니다.
    
    예: "Notice" 단독 → "알아채다" (직역)
        "Notice (document context)" → "공지" (문맥 의역)
    """
    import translators as ts
    import time as _time

    if not texts_to_translate:
        return {}

    translated_map = {}

    # 전체 문서를 훑어서 문맥 힌트를 생성
    full_context = " ".join(texts_to_translate)
    is_document = any(kw in full_context.lower() for kw in [
        'notice', 'welcome', 'please', 'thank', 'contact', 
        'register', 'deadline', 'meeting', 'schedule', 'report',
        'system', 'maintenance', 'inquiry', 'support'
    ])

    for original in texts_to_translate:
        # 짧은 단어(3단어 이하)는 문맥 힌트를 붙여서 번역
        word_count = len(original.split())
        
        if word_count <= 3 and is_document:
            # 문서/공지 맥락의 단어라면 힌트를 붙여서 번역
            hint_text = f"{original} (in a document/notice context)"
        else:
            # 긴 문장은 그대로 번역 (이미 충분한 문맥 포함)
            hint_text = original

        try:
            result = ts.translate_text(hint_text, to_language='ko', translator='bing')
            # 힌트 부분 잔여물 제거 (다양한 괄호 형태 대응)
            result = re.sub(r'\s*[\(\(（].*?문맥.*?[\)\)）]\s*', '', result).strip()
            result = re.sub(r'\s*[\(\(（].*?context.*?[\)\)）]\s*', '', result, flags=re.IGNORECASE).strip()
            result = re.sub(r'\s*[\(\(（].*?[Dd]ocument.*?[\)\)）]\s*', '', result).strip()
            translated_map[original] = result
        except Exception:
            try:
                result = ts.translate_text(hint_text, to_language='ko', translator='google')
                result = re.sub(r'\s*[\(\(（].*?문맥.*?[\)\)）]\s*', '', result).strip()
                result = re.sub(r'\s*[\(\(（].*?context.*?[\)\)）]\s*', '', result, flags=re.IGNORECASE).strip()
                result = re.sub(r'\s*[\(\(（].*?[Dd]ocument.*?[\)\)）]\s*', '', result).strip()
                translated_map[original] = result
            except Exception:
                translated_map[original] = original

        # API 요청 간 짧은 대기 (rate limit 방지)
        _time.sleep(0.3)

    return translated_map


def visualize_and_save(image_path, results, translation_map, font_path="malgun.ttf"):
    """
    원본 이미지 위에 번역된 텍스트를 깔끔하게 덮어쓰는 시각화를 수행합니다.
    """
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"[오류] 이미지를 불러올 수 없습니다: {image_path}")
        return None

    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil, "RGBA")

    # 폰트 로드
    try:
        font_pil = ImageFont.truetype(font_path, 20)
    except IOError:
        font_pil = ImageFont.load_default()
        print("[경고] '맑은 고딕' 폰트를 찾을 수 없어 기본 폰트를 사용합니다.")

    # ─── 콘솔 출력 ──────────────────────────
    print()
    print("╔" + "═"*60 + "╗")
    print("║" + " OCR 인식 및 문맥 기반 번역 결과 ".center(48) + "║")
    print("╠" + "═"*60 + "╣")

    text_report_lines = []
    text_report_lines.append("=" * 60)
    text_report_lines.append("  OCR 인식 및 문맥 기반 번역 결과")
    text_report_lines.append("=" * 60)

    for (bbox, text, prob) in results:
        tl, tr, br, bl = bbox
        tl_x, tl_y = int(tl[0]), int(tl[1])
        br_x, br_y = int(br[0]), int(br[1])

        # 번역 적용 여부 판단
        if text in translation_map:
            translated = translation_map[text]
            display_text = translated
            print(f"║  원문: {text}")
            print(f"║  번역: {translated}  (신뢰도: {prob:.2f})")
            print("║" + "─"*60)
            text_report_lines.append(f"  원문: {text}")
            text_report_lines.append(f"  번역: {translated}  (신뢰도: {prob:.2f})")
            text_report_lines.append("-" * 60)
        else:
            display_text = text
            print(f"║  내용: {text}  (신뢰도: {prob:.2f})")
            print("║" + "─"*60)
            text_report_lines.append(f"  내용: {text}  (신뢰도: {prob:.2f})")
            text_report_lines.append("-" * 60)

        # 시각화: 흰색 박스로 원본 덮기 + 번역 텍스트
        padding = 3
        draw.rectangle(
            [tl_x - padding, tl_y - padding, br_x + padding, br_y + padding],
            fill=(255, 255, 255, 235),
            outline=(200, 200, 200),
            width=1
        )

        # 텍스트를 박스 중앙에 위치시킴
        text_y = tl_y + (br_y - tl_y) // 2 - 12
        draw.text((tl_x + 3, max(0, text_y)), display_text, font=font_pil, fill=(30, 30, 30, 255))

    print("╚" + "═"*60 + "╝")
    text_report_lines.append("=" * 60)

    # 텍스트 결과 파일 저장
    with open(OUTPUT_TEXT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(text_report_lines))
    print(f"\n[저장] 텍스트 결과 → {OUTPUT_TEXT}")

    return img_pil


def main():
    parser = argparse.ArgumentParser(
        description="영한 혼합 문서 OCR + 문맥 기반 번역 + 시각화"
    )
    parser.add_argument(
        '--image', type=str, default="",
        help="분석할 이미지 경로 (입력하지 않으면 기본 샘플 사용)"
    )
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        print("[정보] 입력된 이미지가 없어 기본 샘플을 사용합니다.")
        image_path = DEFAULT_IMAGE_PATH
        download_image(DEFAULT_IMAGE_URL, image_path)

    # ─── OCR 모델 로딩 ───
    print("\n[모델] EasyOCR (ko+en) 로딩 중...")
    t0 = time.time()
    reader = easyocr.Reader(['ko', 'en'], gpu=False)
    print(f"[모델] 로딩 완료 ({time.time()-t0:.1f}초)")

    # ─── OCR 인식 ───
    print(f"[OCR] '{image_path}' 분석 중...")
    t0 = time.time()
    results = reader.readtext(image_path)
    print(f"[OCR] 인식 완료 ({time.time()-t0:.1f}초, {len(results)}개 텍스트 검출)")

    if not results:
        print("[결과] 인식된 텍스트가 없습니다.")
        return

    # ─── 영문 텍스트 추출 및 문맥 기반 번역 ───
    english_texts = []
    for (bbox, text, prob) in results:
        if is_english(text):
            english_texts.append(text)

    print(f"\n[번역] 영문 텍스트 {len(english_texts)}건 감지, 문맥 기반 번역 수행 중...")
    translation_map = {}
    if english_texts:
        translation_map = contextual_translate(english_texts)

    # ─── 시각화 및 저장 ───
    result_img = visualize_and_save(image_path, results, translation_map)

    if result_img:
        result_img.save(OUTPUT_VISUAL)
        print(f"[저장] 시각화 이미지 → {OUTPUT_VISUAL}")

    print("\n[완료] 모든 작업이 성공적으로 완료되었습니다!")


if __name__ == "__main__":
    main()
