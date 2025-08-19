import os, time, glob
import pyautogui as p
from PIL import Image

REGION = (0, 0, 1800, 1168)
PAGES = 290
ADVANCE_KEY = 'space'
DELAY = 0.25

os.makedirs('caps', exist_ok=True)
time.sleep(3)
for i in range(PAGES):
    img = p.screenshot(region=REGION)
    img.save(f'caps/{i:04d}.png')
    p.press(ADVANCE_KEY)
    time.sleep(DELAY)

files = sorted(glob.glob('caps/*.png'))
imgs = [Image.open(f).convert('RGB') for f in files]
imgs[0].save('ebook.pdf', save_all=True, append_images=imgs[1:])
