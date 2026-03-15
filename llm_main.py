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

app = FastAPI(title="연인 궁합 테스트 앱")

# Jinja2 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """메인 페이지를 렌더링합니다."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/compatibility", response_class=HTMLResponse)
async def check_compatibility(
    request: Request,
    p1_name: str = Form(...),
    p1_dob: str = Form(...),
    p1_age: int = Form(...),
    p1_zodiac: str = Form(...),
    p2_name: str = Form(...),
    p2_dob: str = Form(...),
    p2_age: int = Form(...),
    p2_zodiac: str = Form(...)
):
    """LLM을 호출하여 입력된 정보로 연인 궁합을 분석합니다."""
    
    if not api_key:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "환경변수에 GEMINI_API_KEY가 설정되지 않았습니다."
        })
        
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    prompt = f"""
다음 두 사람의 정보를 바탕으로 연인으로서의 궁합을 분석해줘.
궁합 점수(0~100점)를 매기고, 이유를 상세하고 로맨틱하게, 그리고 재미있게 설명해줘.
결과에는 어울리는 이모티콘들도 풍부하게 사용해줘😘

[첫 번째 사람]
- 이름: {p1_name}
- 생년월일: {p1_dob}
- 나이: {p1_age}세
- 별자리: {p1_zodiac}

[두 번째 사람]
- 이름: {p2_name}
- 생년월일: {p2_dob}
- 나이: {p2_age}세
- 별자리: {p2_zodiac}

출력은 반드시 다른 말 없이 아래 JSON 형식으로만 작성할 것. 마크다운 기호(```json)도 쓰지마.
{{
    "점수": 00,
    "궁합_설명": "이유 설명 내용"
}}
"""
    
    try:
        response = model.generate_content(prompt)
        result_text = response.text.strip()
        
        # 혹시 모를 마크다운 찌꺼기 제거
        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
            
        result_data = json.loads(result_text.strip())
        
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "result": result_data,
            "p1_name": p1_name,
            "p2_name": p2_name,
            # 입력 유지용
            "form_data": {
                "p1_name": p1_name, "p1_dob": p1_dob, "p1_age": p1_age, "p1_zodiac": p1_zodiac,
                "p2_name": p2_name, "p2_dob": p2_dob, "p2_age": p2_age, "p2_zodiac": p2_zodiac
            }
        })
    except json.JSONDecodeError as e:
        print(f"JSON 파싱 에러: {response.text}")
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": "LLM 결과 형식을 파싱할 수 없습니다. 다시 시도해주세요."
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "error": f"API 호출 중 오류가 발생했습니다: {e}"
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("llm_main:app", host="127.0.0.1", port=8000, reload=True)
