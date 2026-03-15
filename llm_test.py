import os
import argparse
from dotenv import load_dotenv
import google.generativeai as genai

def classify_sentiment(text):
    # .env 파일에서 환경변수 로드
    load_dotenv()
    
    # 환경변수에서 API 키 가져오기
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("오류: 환경변수에 GEMINI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        return
    
    # API 키 설정
    genai.configure(api_key=api_key)
    
    # 사용할 모델 지정
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    # 프롬프트 구성
    prompt = f"""
주어진 텍스트의 감정을 분석하여 다음 사항들을 포함해 답변해줘:
1. "긍정" 또는 "부정" 분류 결과 (결과에 어울리는 이모티콘을 반드시 포함해줘, 예: 긍정 😊, 부정 😢)
2. 판단에 대한 정확도 (예: 95%)
3. 그렇게 판단한 근거

출력 형식은 아래와 같은 JSON 형식으로만 작성해줘:
{{
    "감정": "긍정/부정 (이모티콘 포함)",
    "정확도": "00%",
    "판단_근거": "이유 설명"
}}

텍스트: "{text}"
"""
    
    try:
        # 모델 호출 및 결과 출력
        response = model.generate_content(prompt)
        print(f"[입력 텍스트]: {text}")
        print(f"\n[분류 결과 및 분석]:\n{response.text.strip()}")
    except Exception as e:
        print(f"API 호출 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # argparse를 사용하여 명령줄 인자 처리
    parser = argparse.ArgumentParser(description="입력한 텍스트의 긍정/부정을 분류하는 스크립트")
    
    # 사용자로부터 입력될 텍스트 인자 추가 (기본값 설정)
    parser.add_argument(
        "-t", "--text", 
        type=str, 
        default="오늘 날씨가 너무 좋아서 기분이 최고야!", 
        help="분류할 텍스트를 입력하세요. (문자열)"
    )
    
    args = parser.parse_args()
    
    # 감정 분류 함수 실행
    classify_sentiment(args.text)
