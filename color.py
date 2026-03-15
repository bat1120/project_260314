import cv2
import numpy as np
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.screenmanager import Screen
from kivymd.app import MDApp
from kivymd.uix.button import MDRaisedButton
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.properties import ObjectProperty, StringProperty, ListProperty

# KV Layout
KV = '''
<ColorMixer>:
    orientation: 'vertical'
    spacing: dp(10)
    padding: dp(20)

    MDLabel:
        text: "색상 스캔 및 혼합 예측"
        halign: "center"
        font_style: "H5"
        size_hint_y: None
        height: self.texture_size[1]

    # Camera Preview
    BoxLayout:
        id: camera_preview
        size_hint_y: 0.5
        canvas.before:
            Color:
                rgba: 0, 0, 0, 1
            Rectangle:
                pos: self.pos
                size: self.size
        Image:
            id: cam_image

    # Target Indicator (Center crosshair)
    MDLabel:
        text: "스캔할 색상을 카메라 중앙에 맞춰주세요"
        halign: "center"
        theme_text_color: "Secondary"
        font_style: "Caption"

    # Color Selection
    MDBoxLayout:
        adaptive_height: True
        spacing: dp(20)
        padding: [dp(10), 0]
        
        MDBoxLayout:
            orientation: 'vertical'
            adaptive_height: True
            MDRaisedButton:
                text: "색상 1 선택"
                on_release: root.select_color(1)
                pos_hint: {"center_x": .5}
            Widget:
                size_hint_y: None
                height: dp(40)
                canvas:
                    Color:
                        rgb: root.color1_rgb
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [dp(10)]

        MDBoxLayout:
            orientation: 'vertical'
            adaptive_height: True
            MDRaisedButton:
                text: "색상 2 선택"
                on_release: root.select_color(2)
                pos_hint: {"center_x": .5}
            Widget:
                size_hint_y: None
                height: dp(40)
                canvas:
                    Color:
                        rgb: root.color2_rgb
                    RoundedRectangle:
                        pos: self.pos
                        size: self.size
                        radius: [dp(10)]

    # Result
    MDBoxLayout:
        orientation: 'vertical'
        adaptive_height: True
        spacing: dp(10)
        
        MDLabel:
            text: "혼합 결과 예측"
            halign: "center"
            font_style: "H6"

        Widget:
            size_hint_y: None
            height: dp(60)
            canvas:
                Color:
                    rgb: root.mixed_color_rgb
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [dp(15)]
        
        MDLabel:
            text: root.mixing_info
            halign: "center"
            theme_text_color: "Hint"
'''

class ColorMixer(MDBoxLayout):
    color1_rgb = ListProperty([0.5, 0.5, 0.5])
    color2_rgb = ListProperty([0.5, 0.5, 0.5])
    mixed_color_rgb = ListProperty([0.5, 0.5, 0.5])
    mixing_info = StringProperty("두 색상을 선택하면 결과가 나타납니다.")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0/30.0)
        self.current_scan_rgb = (0, 0, 0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Get center color
            h, w, _ = frame.shape
            center_pixel = frame[h//2, w//2] # BGR
            self.current_scan_rgb = (center_pixel[2]/255.0, center_pixel[1]/255.0, center_pixel[0]/255.0)

            # Draw target indicator
            cv2.circle(frame, (w//2, h//2), 10, (0, 255, 0), 2)
            
            # Convert to texture
            buf1 = cv2.flip(frame, 0)
            buf = buf1.tostring()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.ids.cam_image.texture = texture

    def select_color(self, slot):
        if slot == 1:
            self.color1_rgb = self.current_scan_rgb
        else:
            self.color2_rgb = self.current_scan_rgb
        self.calculate_mixing()

    def calculate_mixing(self):
        # 감색 혼합 (Subtractive Mixing) 단순 구현
        # CMY 색상 공간에서 섞는 방식으로 시뮬레이션
        c1 = [1 - x for x in self.color1_rgb]
        c2 = [1 - x for x in self.color2_rgb]
        
        # 실제 물감 혼합은 평균보다 조금 더 어두워지는 경향이 있음
        mixed_cmy = [(a + b) / 2 for a, b in zip(c1, c2)]
        
        # 다시 RGB로 변환
        self.mixed_color_rgb = [1 - x for x in mixed_cmy]
        self.mixing_info = f"혼합 완료! (R:{int(self.mixed_color_rgb[0]*255)}, G:{int(self.mixed_color_rgb[1]*255)}, B:{int(self.mixed_color_rgb[2]*255)})"

class ColorApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"
        Builder.load_string(KV)
        return ColorMixer()

    def on_stop(self):
        # 카메라 해제
        self.root.capture.release()

if __name__ == '__main__':
    ColorApp().run()
