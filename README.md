# eBook 캡처 · 업스케일 · PDF 생성 도구

스크린에서 페이지를 연속 캡처하고(예: 뷰어 `space`로 페이지 넘김), Real-ESRGAN으로 고해상도 업스케일한 뒤, 인쇄용/확대용으로 품질 좋은 PDF를 만드는 파이프라인입니다.

* 캡처 → `caps/*.png`
* 업스케일 → `caps_upscaled/*.png` (또는 원본 덮어쓰기)
* PDF 생성 → `ebook.pdf` (JPEG 변환 + 무손실 레이아웃)

---

## 주요 기능

* 지정 영역 연속 캡처 및 자동 페이지 넘김 (`pyautogui`)
* Real-ESRGAN(SRVGG) 기반 x4 업스케일(일반/디노이즈/애니 전용 모델)
* JPEG(퀄리티/서브샘플링 제어) → PDF 병합(`img2pdf`)
* 대량 처리 안정성: 타일, 임시 파일, 재시도 스킵 등

---

## 요구 사항

* Python 3.9+
* macOS/Windows/Linux
* (선택) NVIDIA GPU + CUDA (있으면 자동 사용)

### 권한/환경

* **macOS**: `pyautogui`로 캡처하려면 시스템 설정 → 개인정보 보호 및 보안 → **화면 기록** 권한을 터미널/IDE에 부여
* **뷰어 단축키**: 기본은 `space`로 다음 페이지. 다른 키라면 아래 캡처 스크립트의 `ADVANCE_KEY`를 바꿔주세요.

---

## 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip

pip install pyautogui pillow tqdm numpy img2pdf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 환경
# CPU만 쓸 경우: pip install torch torchvision

pip install realesrgan
```

> Real-ESRGAN 가중치는 스크립트가 자동 다운로드합니다(실패 시 수동 배치).

---

## 폴더 구조(권장)

```
project/
 ├─ capture_and_make_pdf.py     # 캡처 + PDF 간단 파이프라인
 ├─ upscale_cli.py              # 업스케일 배치 CLI
 ├─ pdf_make.py                 # JPG → PDF 병합(고품질)
 ├─ caps/                       # 캡처 원본 PNG
 ├─ caps_upscaled/              # 업스케일 결과
 └─ .weights/realesrgan/        # (자동) 모델 가중치
```

> 아래 README의 스니펫은 사용자가 올린 코드와 동일/동등 기능을 수행합니다. 파일 이름은 자유롭게 분리해도 됩니다.

---

## 1) 화면 캡처 & 1차 PDF 만들기

`capture_and_make_pdf.py` 예시:

```python
import os, time, glob
import pyautogui as p
from PIL import Image

REGION = (0, 0, 1800, 1168)   # 캡처 영역(x, y, w, h)
PAGES = 290                   # 캡처할 페이지 수
ADVANCE_KEY = 'space'         # 페이지 넘김 키
DELAY = 0.25                  # 캡처 간 대기(초)

os.makedirs('caps', exist_ok=True)
time.sleep(3)                 # 뷰어로 포커스 이동할 시간 - 3초 안에 화면 전환해두시면 됩니다

for i in range(PAGES):
    img = p.screenshot(region=REGION)
    img.save(f'caps/{i:04d}.png')
    p.press(ADVANCE_KEY)
    time.sleep(DELAY)

# (선택) 원본 PNG들을 그냥 PDF로 합치기 — 품질 개선 전 확인용
files = sorted(glob.glob('caps/*.png'))
imgs = [Image.open(f).convert('RGB') for f in files]
imgs[0].save('ebook.pdf', save_all=True, append_images=imgs[1:])
```
**캡처 영역(REGION) 설정 방법**  
스크립트의 `REGION = (0, 0, 1800, 1168)` 부분은 캡처할 화면 영역을 지정하는 값입니다.
형식은 `(x, y, width, height)`  

`pip install pyautogui` 이후, 
1. 먼저 아래 명령어를 실행
    
    ```bash
    python -c "import pyautogui as p; print(p.position())"
    ```
    
2. 실행하면 터미널에서 **마우스 포인터 좌표**를 실시간으로 확인할 수 있습니다.
3. 캡처하려는 영역의 **왼쪽 상단 좌표**와 **오른쪽 하단 좌표**를 각각 마우스를 올려 확인하세요.
4. `REGION` 값은 다음과 같이 계산
    
    ```
    REGION = (왼쪽상단_x, 왼쪽상단_y, 오른쪽하단_x - 왼쪽상단_x, 오른쪽하단_y - 왼쪽상단_y)
    ```

**팁**

* 흐릿하면 `REGION`을 더 크게 잡거나, 뷰어의 실제 렌더 배율(줌)을 올린 뒤 캡처하세요.
* 느리면 `DELAY`를 약간 늘려 페이지 전환/렌더 대기 시간을 확보하세요.

---

## 2) Real-ESRGAN 업스케일

`upscale_cli.py` (배치 CLI):

```python
import argparse, os, sys
from pathlib import Path
from urllib.request import urlopen
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact

WEIGHTS = {
    "general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    "general-wdn-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    "animevideov3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def download_if_needed(url: str, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    try:
        with urlopen(url) as r:
            data = r.read()
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
        return dst
    except Exception as e:
        print(f"[WARN] 자동 다운로드 실패: {url}\n       → {e}\n       수동으로 내려받아 {dst}에 두면 됩니다.", file=sys.stderr)
        return dst if dst.exists() else None

def build_upsampler(device: str, model_key: str, weights_dir: Path, tile: int, half: bool):
    if model_key not in WEIGHTS:
        raise ValueError("model must be one of: general-x4v3, general-wdn-x4v3, animevideov3")

    weights_path = weights_dir / (Path(WEIGHTS[model_key]).name)
    ok = download_if_needed(WEIGHTS[model_key], weights_path)
    if ok is None:
        raise RuntimeError("모델 가중치가 없습니다. 링크로 직접 내려받아 weights 폴더에 두세요.")

    dev = "cuda" if device == "auto" and torch.cuda.is_available() else device
    if dev == "auto":
        dev = "cpu"

    if model_key == "animevideov3":
        net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
    else:
        net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")

    upsampler = RealESRGANer(
        scale=4,
        model_path=str(weights_path),
        model=net,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=half if dev == "cuda" else False,
        device=dev
    )
    return upsampler

def iter_images(root: Path, recursive: bool):
    it = root.rglob("*") if recursive else root.iterdir()
    for p in it:
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p

def load_image(path: Path):
    img = Image.open(path)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    return img

def safe_save(image: Image.Image, out_path: Path, quality: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        image.save(out_path, format="JPEG", quality=95 if quality is None else quality, subsampling=0)
    elif ext == ".png":
        image.save(out_path, format="PNG")
    elif ext == ".webp":
        image.save(out_path, format="WEBP", quality=95 if quality is None else quality, method=6)
    elif ext in {".tif", ".tiff"}:
        image.save(out_path, format="TIFF", compression="tiff_deflate")
    else:
        image.save(out_path)

def main():
    ap = argparse.ArgumentParser(description="Batch upscale images with Real-ESRGAN (latest SRVGG models)")
    ap.add_argument("--input_dir", required=True, help="입력 폴더")
    ap.add_argument("--output_dir", default=None, help="출력 폴더 (미지정 시 입력폴더명 + _upscaled)")
    ap.add_argument("--model", default="general-x4v3",
                    choices=["general-x4v3","general-wdn-x4v3","animevideov3"],
                    help="모델 선택 (일반/디노이즈/애니)")
    ap.add_argument("--device", default="auto", choices=["auto","cuda","cpu"])
    ap.add_argument("--recursive", action="store_true", help="하위 폴더까지 처리")
    ap.add_argument("--inplace", action="store_true", help="원본 위에 덮어쓰기")
    ap.add_argument("--quality", type=int, default=95, help="JPEG/WEBP 저장 퀄리티")
    ap.add_argument("--weights_dir", default=".weights/realesrgan", help="가중치 저장 폴더")
    ap.add_argument("--tile", type=int, default=256, help="타일 크기(메모리 절약). 0=비활성")
    ap.add_argument("--half", action="store_true", help="CUDA half-precision 사용")
    ap.add_argument("--outscale", type=float, default=4.0, help="최종 배율(기본 4배, 1~4 권장)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        print("입력 폴더를 확인하세요.", file=sys.stderr); sys.exit(1)

    output_dir = input_dir if args.inplace else Path(args.output_dir or (str(input_dir) + "_upscaled")).resolve()
    weights_dir = Path(args.weights_dir).expanduser().resolve()

    upsampler = build_upsampler(args.device, args.model, weights_dir, tile=args.tile, half=args.half)

    files = list(iter_images(input_dir, args.recursive))
    if not files:
        print("처리할 이미지가 없습니다."); return

    pbar = tqdm(files, ncols=100, desc=f"Upscaling ({args.model})")
    for src in pbar:
        rel = src.relative_to(input_dir)
        dst = src if args.inplace else (output_dir / rel)
        tmp = dst.parent / (dst.stem + ".tmp" + dst.suffix)

        try:
            img = load_image(src)
            sr_np, _ = upsampler.enhance(np.array(img), outscale=args.outscale)
            out = Image.fromarray(sr_np)
            safe_save(out, tmp, quality=args.quality)
            os.replace(tmp, dst)
        except Exception as e:
            try:
                if tmp.exists(): tmp.unlink()
            except Exception:
                pass
            tqdm.write(f"[SKIP] {src} -> {e}")

    print(f"완료: {len(files)}개 이미지 처리됨")
    if not args.inplace:
        print(f"출력 폴더: {output_dir}")

if __name__ == "__main__":
    main()
```

### 사용 예

```bash
# caps 폴더의 PNG들을 4배 업스케일하여 caps_upscaled에 저장
python upscale_cli.py --input_dir caps

# 원본 위 덮어쓰기(용량 줄이려면 PNG→JPG로 변환은 다음 단계에서)
python upscale_cli.py --input_dir caps --inplace

# 디노이즈 강한 모델 + 작은 타일(메모리 부족 대비) + CPU 강제
python upscale_cli.py --input_dir caps --model general-wdn-x4v3 --tile 128 --device cpu

# 애니/일러스트 전용 모델 + half-precision(CUDA)
python upscale_cli.py --input_dir caps --model animevideov3 --half
```

**파라미터 가이드**

* `--model`

  * `general-x4v3`: 일반 사진/스캔 권장
  * `general-wdn-x4v3`: 노이즈 많은 스캔/저화질
  * `animevideov3`: 애니/일러스트/라인아트
* `--tile`: OOM 방지. 128\~256 권장. 너무 작으면 경계 인접 아티팩트 가능 → `tile_pad`가 완화
* `--outscale`: 결과 배율. 1\~4 권장(모델은 x4 학습). 2\~3배면 용량-품질 균형
* `--device`: `auto`(CUDA 있으면 사용), `cuda`, `cpu`
* `--half`: CUDA에서 VRAM 절약/속도 향상

---

## 3) 고품질 PDF 생성

업스케일된 PNG를 **무손실 레이아웃**으로 묶기 위해, JPEG로 변환(4:4:4, `subsampling=0`) 후 PDF로 합칩니다.

`pdf_make.py`:

```python
from PIL import Image
import glob, img2pdf

INPUT_GLOB = "caps_upscaled/*.png"
OUTPUT_PDF = "ebook.pdf"

jpg_files = []
for f in sorted(glob.glob(INPUT_GLOB)):
    img = Image.open(f).convert("RGB")
    out_f = f.replace(".png", ".jpg")
    img.save(out_f, "JPEG", quality=92, optimize=True)  # 92~95 권장
    jpg_files.append(out_f)

with open(OUTPUT_PDF, "wb") as f:
    f.write(img2pdf.convert(jpg_files))
```

### 사용 예

```bash
python pdf_make.py
# → ebook.pdf 생성
```

**품질 팁**

* **품질/용량 균형**: `quality=92`는 맥/PC 확대 감상용으로 충분한 경우가 많습니다. 더 선명하게는 `95`.
* **서브샘플링 0**: 위 업스케일 단계에서 JPEG 저장 시 이미 `subsampling=0` 처리. 컬러/텍스트 가장자리 보존에 유리.
* **PDF DPI**: `img2pdf`는 픽셀-포인트 매핑으로 실제 픽셀을 그대로 박아넣습니다. 업스케일을 충분히 하면 추가 DPI 조절 없이도 선명합니다.

---

## 빠른 시작(요약)

```bash
# 0) 가상환경 + 의존성 설치
python -m venv .venv && source .venv/bin/activate
pip install pyautogui pillow tqdm numpy img2pdf realesrgan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA

# 1) 뷰어 포커스 후 캡처 시작(키: space, 총 PAGES/REGION 수정)
python capture_and_make_pdf.py

# 2) 업스케일(일반 x4, caps → caps_upscaled)
python upscale_cli.py --input_dir caps

# 3) PDF 생성(품질 92의 JPG로 변환 후 병합)
python pdf_make.py
```

---

## 문제 해결

* **macOS에서 캡처가 안 됨**: 화면 기록/손쉬운 사용 권한 확인. 외부 모니터일 경우 `REGION` 좌표 재확인.
* **GPU 미검출**: `torch.cuda.is_available()`가 False면 CPU로 동작. CUDA 휠/드라이버 버전 일치 점검.
* **메모리 부족(OOM)**: `--tile`을 더 작게(128/64) → 속도는 느려지지만 메모리 사용량 감소.
* **가중치 다운로드 실패**: 로그에 표시된 경로로 수동 저장(예: `.weights/realesrgan/realesr-general-x4v3.pth`).

---

## 추가 팁

* 맥/PC **확대 감상**이 목적이면, 업스케일 x4 후 PDF 내 JPEG 품질 **92\~95**가 체감상 괜찮았습니다.
* 스캔생성물/도트/라인이 많은 자료는 `general-wdn-x4v3` 또는 `animevideov3`를 비교해보세요.
* 컬러 밴딩/텍스트 헤일로가 보이면

  1. 업스케일 배율을 2~3으로 낮추거나,
  2. JPEG `quality`를 95로 올려 보세요.
* 페이지 전환이 느린 뷰어는 `DELAY`를 0.35\~0.5로 늘려 누락 방지

---

본 자료는 개인적 열람 및 학습 목적을 위해 제작된 것입니다.  
저작권자의 허가 없이 무단으로 복제, 배포, 공유, 판매하는 행위는 저작권법에 따라 법적 처벌을 받을 수 있습니다.  
