import os
import argparse
from dotenv import load_dotenv
import google.generativeai as genai

def rewrite_email(text):
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
당신은 10년 차 상위 1% 비즈니스 커뮤니케이션 전문가입니다.
사용자가 감정적으로 격양되거나 다소 공격적/비공식적으로 작성한 아래의 텍스트를, 
매우 정중하고 프로페셔널하며 상대방의 기분을 상하게 하지 않으면서도 
원하는 바(요청, 불만 제기, 독촉 등)를 명확하게 전달하는 '비즈니스 이메일'로 변환해 주세요.

[요구사항]
1. 적절한 이메일 제목을 하나 추천해주세요.
2. 비즈니스 환경에 맞는 정중한 인사말과 맺음말을 포함해주세요.
3. 감정적인 표현은 제거하고, 사실 기반과 협업을 강조하는 톤으로 바꿔주세요.

원본 텍스트: "{text}"
"""
    
    try:
        # 모델 호출 및 결과 출력
        print("⏳ 정중한 비즈니스 이메일로 변환 중입니다...\n")
        response = model.generate_content(prompt)
        
        print("=" * 60)
        print("[🤬 원본 텍스트]")
        print(text)
        print("=" * 60)
        print("[✉️ 변환된 비즈니스 이메일]")
        print(response.text.strip())
        print("=" * 60)
        
    except Exception as e:
        print(f"API 호출 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # argparse를 사용하여 명령줄 인자 처리
    parser = argparse.ArgumentParser(description="감정적인 텍스트를 정중한 비즈니스 이메일로 변환해주는 스크립트")
    
    # 사용자로부터 입력될 텍스트 인자 추가 (기본값으로는 다소 화가 난 텍스트 설정)
    parser.add_argument(
        "-t", "--text", 
        type=str, 
        default="아니 이거 저번에 해달라고 한지가 언젠데 아직도 안해주고 뭐하는거에요? 내일까지 당장 안 주면 진짜 큰일 날 줄 아세요. 아 짜증나.", 
        help="변환할 메시지 내용을 입력하세요."
    )
    
    args = parser.parse_args()
    
    # 변환 함수 실행
    rewrite_email(args.text)
