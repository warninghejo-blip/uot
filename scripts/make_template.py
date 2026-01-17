import os
from PIL import Image, ImageDraw, ImageFont

os.makedirs('templates', exist_ok=True)
path = os.path.join('templates', 'bull.jpg')
img = Image.new('RGB', (900, 600), (10, 10, 30))
draw = ImageDraw.Draw(img)
text = 'FENNEC vs BEAR'
try:
    font = ImageFont.truetype('arial.ttf', 60)
except Exception:
    font = ImageFont.load_default()
bbox = draw.textbbox((0, 0), text, font=font)
width = bbox[2] - bbox[0]
height = bbox[3] - bbox[1]
draw.text(((900 - width) / 2, 80), text, fill=(255, 180, 0), font=font)
draw.rectangle([150, 250, 350, 500], outline=(255, 80, 80), width=5)
draw.rectangle([550, 200, 780, 520], outline=(80, 255, 120), width=5)
img.save(path, 'JPEG', quality=90)
print(f"Created template at {path}")
