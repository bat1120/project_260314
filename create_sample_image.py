from PIL import Image, ImageDraw, ImageFont

def create_sample():
    lines = [
        "[ Notice / 공지사항 ]",
        "",
        "Welcome to the AI Research Lab.",
        "인공지능 연구실에 오신 것을 환영합니다.",
        "",
        "All participants must register before the deadline.",
        "모든 참가자는 마감일 전에 등록해야 합니다.",
        "",
        "The system will be down for maintenance tonight.",
        "Please save your work and log out by 6 PM.",
        "",
        "For inquiries, contact support@example.com",
        "문의사항은 위 이메일로 연락 바랍니다.",
        "",
        "Thank you for your cooperation.",
    ]

    # 이미지 사이즈 결정
    img = Image.new('RGB', (900, 480), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("malgun.ttf", 22)
    except IOError:
        font = ImageFont.load_default()

    y = 25
    for line in lines:
        d.text((35, y), line, fill=(0, 0, 0), font=font)
        y += 30

    img.save('mixed_text_sample.png')
    print("혼합 텍스트 샘플 이미지 생성 완료: mixed_text_sample.png")

if __name__ == "__main__":
    create_sample()
