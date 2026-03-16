import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

class GeminiAI:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        # Using gemini-1.5-flash for fast and efficient chat
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.chat = self.model.start_chat(history=[])

    def get_response(self, user_input):
        try:
            response = self.chat.send_message(user_input)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

if __name__ == "__main__":
    # Simple test
    ai = GeminiAI()
    print("AI: Hello! How can I help you today?")
    while True:
        user = input("You: ")
        if user.lower() in ['exit', 'quit']:
            break
        print(f"AI: {ai.get_response(user)}")
