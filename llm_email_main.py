import os
import json
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import google.generativeai as genai

# 환경변수 로드
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

app = FastAPI(title="비즈니스 이메일 순화기")

# Jinja2 템플릿 디렉토리 설정 (기존 templates 폴더 사용)
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지를 렌더링합니다."""
    return templates.TemplateResponse("email_index.html", {"request": request})

@app.post("/api/rewrite", response_class=HTMLResponse)
async def rewrite_email(
    request: Request,
    original_text: str = Form(...)
):
    """LLM을 호출하여 입력된 감정적인 텍스트를 정중한 비즈니스 이메일로 변환합니다."""
    
    if not api_key:
        return templates.TemplateResponse("email_index.html", {
            "request": request, 
            "error": "환경변수에 GEMINI_API_KEY가 설정되지 않았습니다."
        })
        
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    prompt = f"""
당신은 10년 차 상위 1% 비즈니스 커뮤니케이션 전문가입니다.
사용자가 감정적으로 격양되거나 다소 공격적/비공식적으로 작성한 아래의 텍스트를, 
매우 정중하고 프로페셔널하며 상대방의 기분을 상하게 하지 않으면서도 
원하는 바(요청, 불만 제기, 독촉 등)를 명확하게 전달하는 '비즈니스 이메일'로 변환해 주세요.

[요구사항]
1. 적절한 이메일 제목을 하나 추천해주세요.
2. 비즈니스 환경에 맞는 정중한 인사말과 맺음말을 포함해주세요.
3. 감정적인 표현은 제거하고, 사실 기반과 협업을 강조하는 톤으로 바꿔주세요.

원본 텍스트: "{original_text}"

출력은 반드시 다른 말 없이 아래 JSON 형식으로만 작성할 것. 마크다운 기호(```json 등)도 쓰지마.
{{
    "제목": "추천하는 이메일 제목",
    "본문": "정중하게 변환된 이메일 본문 (줄바꿈 포함)"
}}
"""
    
    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # 마크다운 찌꺼기 제거 처리
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
            
        result_data = json.loads(result_text.strip())
        
        return templates.TemplateResponse("email_index.html", {
            "request": request, 
            "result": result_data,
            "original_text": original_text
        })
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 에러: {response.text}")
        return templates.TemplateResponse("email_index.html", {
            "request": request, 
            "error": "LLM 결과 형식을 파싱할 수 없습니다. 다시 시도해주세요.",
            "original_text": original_text
        })
    except Exception as e:
        return templates.TemplateResponse("email_index.html", {
            "request": request, 
            "error": f"API 호출 중 오류가 발생했습니다: {e}",
            "original_text": original_text
        })

if __name__ == "__main__":
    import uvicorn
    # 연인 궁합 앱이 8000포트를 사용할 수 있으므로 8001을 씁니다.
    uvicorn.run("llm_email_main:app", host="127.0.0.1", port=8001, reload=True)
