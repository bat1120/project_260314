from kivymd.app import MDApp
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.properties import StringProperty
from kivymd.uix.card import MDCard
from kivy.clock import Clock
from gemini_ai import GeminiAI
import threading

# Window size for desktop testing
Window.size = (360, 640)

KV = '''
<ChatMessage>:
    orientation: "vertical"
    padding: "12dp"
    size_hint_x: 0.8
    size_hint_y: None
    height: self.minimum_height
    pos_hint: {"right": 1} if root.is_user else {"left": 1}
    md_bg_color: [0.12, 0.58, 0.95, 1] if root.is_user else [0.25, 0.25, 0.25, 1]
    radius: [20, 20, 2, 20] if root.is_user else [20, 20, 20, 2]
    elevation: 2

    MDLabel:
        text: root.text
        theme_text_color: "Custom"
        text_color: [1, 1, 1, 1]
        adaptive_height: True
        font_style: "Body1"

MDScreen:
    md_bg_color: [0.1, 0.1, 0.1, 1]

    MDBoxLayout:
        orientation: "vertical"

        MDTopAppBar:
            title: "AI Chat Assistant"
            elevation: 4
            md_bg_color: [0.15, 0.15, 0.15, 1]
            specific_text_color: [1, 1, 1, 1]
            left_action_items: [["robot-glow", lambda x: None]]

        ScrollView:
            id: chat_scroll
            do_scroll_x: False
            
            MDBoxLayout:
                id: chat_list
                orientation: "vertical"
                padding: "15dp"
                spacing: "20dp"
                size_hint_y: None
                height: self.minimum_height

        MDBoxLayout:
            adaptive_height: True
            padding: "10dp"
            spacing: "10dp"
            md_bg_color: [0.15, 0.15, 0.15, 1]

            MDTextField:
                id: user_input
                hint_text: "메시지를 입력하세요..."
                mode: "round"
                fill_color_normal: [0.2, 0.2, 0.2, 1]
                hint_text_color_normal: [0.7, 0.7, 0.7, 1]
                text_color_normal: [1, 1, 1, 1]
                on_text_validate: app.send_message()

            MDIconButton:
                icon: "send"
                theme_icon_color: "Custom"
                icon_color: [0.12, 0.58, 0.95, 1]
                on_release: app.send_message()
'''

class ChatMessage(MDCard):
    text = StringProperty()
    is_user = StringProperty()

class AIChatApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"
        self.ai = GeminiAI()
        return Builder.load_string(KV)

    def send_message(self):
        text = self.root.ids.user_input.text.strip()
        if text:
            # Add user message
            self.add_message(text, True)
            self.root.ids.user_input.text = ""
            
            # Start AI response thread
            threading.Thread(target=self.get_ai_response, args=(text,)).start()

    def get_ai_response(self, text):
        response = self.ai.get_response(text)
        Clock.schedule_once(lambda dt: self.add_message(response, False))

    def add_message(self, text, is_user):
        msg = ChatMessage(text=text, is_user="True" if is_user else "")
        self.root.ids.chat_list.add_widget(msg)
        
        # Scroll to bottom
        Clock.schedule_once(self.scroll_bottom, 0.1)

    def scroll_bottom(self, dt):
        self.root.ids.chat_scroll.scroll_y = 0

if __name__ == "__main__":
    AIChatApp().run()
