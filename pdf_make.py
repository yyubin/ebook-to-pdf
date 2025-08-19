from PIL import Image
import glob, img2pdf

INPUT_GLOB = "caps_upscaled/*.png"
OUTPUT_PDF = "ebook.pdf"

jpg_files = []
for f in sorted(glob.glob(INPUT_GLOB)):
    img = Image.open(f).convert("RGB")
    out_f = f.replace(".png", ".jpg")
    img.save(out_f, "JPEG", quality=92, optimize=True)
    jpg_files.append(out_f)

with open(OUTPUT_PDF, "wb") as f:
    f.write(img2pdf.convert(jpg_files))
