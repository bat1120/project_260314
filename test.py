import ollama
import sys

def main():
    try:
        print("Ollama에 설치된 모델 목록을 확인합니다...")
        response_list = ollama.list()
        
        models = []
        if hasattr(response_list, 'models'):
            models = response_list.models
        elif isinstance(response_list, dict):
            models = response_list.get('models', [])
        
        if not models:
            print("설치된 모델을 찾을 수 없습니다.")
            return

        model_names = []
        for m in models:
            if hasattr(m, 'model'):
                model_names.append(m.model)
            elif isinstance(m, dict):
                model_names.append(m.get('name') or m.get('model'))
        
        target_model = None
        for name in model_names:
            if 'qwen' in name.lower():
                target_model = name
                break
        
        if not target_model:
            target_model = model_names[0]
            
        print(f"[{target_model}] 모델과 대화를 시작합니다. (종료하려면 'exit' 또는 'quit' 입력)")
        print("-" * 50)

        # 대화 기록 유지
        messages = []

        while True:
            user_input = input("\n나: ")
            
            if user_input.lower() in ['exit', 'quit', 'exit()', 'quit()', '종료']:
                print("대화를 종료합니다.")
                break
                
            if not user_input.strip():
                continue

            messages.append({'role': 'user', 'content': user_input})

            print(f"\n{target_model}: ", end="", flush=True)
            
            try:
                # 스트리밍 응답으로 더 생동감 있게 구현
                response_content = ""
                stream = ollama.chat(
                    model=target_model,
                    messages=messages,
                    stream=True,
                )

                for chunk in stream:
                    content = ""
                    if hasattr(chunk, 'message'):
                        content = chunk.message.content
                    else:
                        content = chunk['message']['content']
                    
                    print(content, end="", flush=True)
                    response_content += content
                
                print() # 줄바꿈
                messages.append({'role': 'assistant', 'content': response_content})
                
            except Exception as e:
                print(f"\n응답 중 에러 발생: {e}")

    except Exception as e:
        print(f"에러 발생: {e}")

if __name__ == "__main__":
    main()