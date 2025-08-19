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
    # 일반 사진용
    "general-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
    # 일반+디노이즈
    "general-wdn-x4v3": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
    # 애니/일러스트
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

    # SRVGGNetCompact 설정: animevideov3는 conv=16, general-x4v3는 conv=32
    if model_key == "animevideov3":
        net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type="prelu")
    else:
        net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type="prelu")

    upsampler = RealESRGANer(
        scale=4,                # 모델 자체는 x4
        model_path=str(weights_path),
        model=net,
        tile=tile,              # 메모리 부족하면 128~256 권장
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
        image.save(out_path, format="JPEG", quality=quality, subsampling=0)
    elif ext == ".png":
        image.save(out_path, format="PNG")
    elif ext == ".webp":
        image.save(out_path, format="WEBP", quality=quality, method=6)
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
    ap.add_argument("--outscale", type=float, default=4.0, help="최종 배율(기본 4배, 1~4 사이 권장)")
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
