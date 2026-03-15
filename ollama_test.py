import requests
import json
import sys
import time
import argparse
import base64
import os

# Windows에서 출력 인코딩을 UTF-8로 설정
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def encode_image_to_base64(image_path):
    """이미지 파일을 base64 문자열로 변환합니다."""
    if not image_path or not os.path.exists(image_path):
        return None
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"이미지 인코딩 에러: {e}")
        return None

def call_ollama_api(messages, model, images=None):
    url = "http://localhost:11434/api/chat"
    
    # 메시지 복사 (원본 보존)
    current_messages = [msg.copy() for msg in messages]
    
    # 마지막 사용자 메시지에 이미지 추가 (있을 경우)
    if images and current_messages:
        # Ollama API는 images 필드를 메시지 객체 내에 기대함
        for i in range(len(current_messages)-1, -1, -1):
            if current_messages[i]["role"] == "user":
                current_messages[i]["images"] = images
                break
    
    payload = {
        "model": model,
        "messages": current_messages,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        end_time = time.time()
        
        result = response.json()
        message = result.get("message", {})
        answer = message.get("content", "")
        elapsed_time = end_time - start_time
        
        return answer, elapsed_time, result
    except requests.exceptions.ConnectionError:
        print("\n에러: Ollama 서버에 연결할 수 없습니다. Ollama가 실행 중인지 확인해주세요.")
        return None, 0, None
    except Exception as e:
        print(f"\n에러 발생: {e}")
        return None, 0, None

def chat_with_ollama(model):
    messages = []
    
    print(f"Ollama 채팅 시작 (모델: {model})")
    print("종료하려면 'exit', 'quit' 또는 'bye'를 입력하세요.\n")
    
    while True:
        try:
            user_input = input("나: ").strip()
            if not user_input: continue
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("채팅을 종료합니다. 안녕히 가세요!")
                break
            
            messages.append({"role": "user", "content": user_input})
            answer, elapsed, full_res = call_ollama_api(messages, model)
            
            if answer is not None:
                print(f"\nAI: {answer if answer else '(응답이 비어있음)'}")
                print(f"[응답 소요 시간: {elapsed:.2f}초]\n")
                if answer:
                    messages.append({"role": "assistant", "content": answer})
                else:
                    print("--- Debug: Full Response ---")
                    print(json.dumps(full_res, indent=2, ensure_ascii=False))
                    print("---------------------------\n")
            else:
                break
                
        except KeyboardInterrupt:
            print("\n채팅을 종료합니다.")
            break

def main():
    parser = argparse.ArgumentParser(description="Ollama API 테스트 및 채팅 스크립트 (VLM 지원)")
    parser.add_argument("--text", type=str, help="Ollama에게 보낼 텍스트")
    parser.add_argument("--image_path", type=str, help="분석할 이미지 파일 경로")
    parser.add_argument("--model", type=str, default="qwen3.5:0.8b", help="사용할 모델명 (VLM인 경우 vision 지원 모델 필요)")
    args = parser.parse_args()

    # 이미지 base64 인코딩
    images = []
    if args.image_path:
        base64_image = encode_image_to_base64(args.image_path)
        if base64_image:
            images.append(base64_image)
            print(f"이미지 로드 완료: {args.image_path}")

    if args.text:
        # 단건 실행 모드
        messages = [{"role": "user", "content": args.text}]
        print(f"[{args.model}] 분석 중: {args.text}")
        answer, elapsed, _ = call_ollama_api(messages, args.model, images)
        
        if answer:
            print(f"\nAI: {answer}")
            print(f"[응답 소요 시간: {elapsed:.2f}초]")
    elif images:
        # 텍스트 없이 이미지만 있는 경우 기본 프롬프트 제공
        messages = [{"role": "user", "content": "이 이미지에 대해 설명해줘."}]
        print(f"[{args.model}] 이미지 분석 중...")
        answer, elapsed, _ = call_ollama_api(messages, args.model, images)
        if answer:
            print(f"\nAI: {answer}")
            print(f"[응답 소요 시간: {elapsed:.2f}초]")
    else:
        # 인터랙티브 채팅 모드
        chat_with_ollama(args.model)

if __name__ == "__main__":
    main()
