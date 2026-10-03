#!/usr/bin/env python3
"""Evaluate a trained 9-1-5 SRCNN on ALWI_002..ALWI_021.

By default, expects srcnn_x4.pth and "Breast image and its masks.zip" beside
this script. Alternatively, supply --data-root pointing at a directory with
images/ and masks/ subdirectories. Requires Python, PyTorch, NumPy, Pillow,
and SciPy. Writes per-case metrics to srcnn_per_case_metrics.csv.
"""
from __future__ import annotations

import argparse
import io
import math
import os
import sys
import zipfile
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError as exc:
    raise SystemExit("PyTorch is required to load srcnn_x4.pth; install torch first.") from exc

CASES = [f"ALWI_{i:03d}" for i in range(2, 22)]
EMPTY_MASK_CASES = {"ALWI_007", "ALWI_008", "ALWI_010", "ALWI_012"}


class SRCNN(nn.Module):
    """SRCNN 9-1-5; explicit reflection padding before each spatial conv."""
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=0)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=0)

    def forward(self, x):
        x = F.pad(x, (4, 4, 4, 4), mode="reflect")
        x = F.relu(self.conv1(x), inplace=False)
        x = F.relu(self.conv2(x), inplace=False)
        x = F.pad(x, (2, 2, 2, 2), mode="reflect")
        return self.conv3(x)


def load_model(weights_path: Path, device: torch.device) -> SRCNN:
    # weights_only is used where supported; older PyTorch versions lack this kwarg.
    try:
        checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(weights_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        for container_key in ("state_dict", "model_state_dict", "model", "net"):
            if container_key in checkpoint and isinstance(checkpoint[container_key], dict):
                checkpoint = checkpoint[container_key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a PyTorch state_dict.")
    # Accept common wrappers such as module.conv1.weight.
    state = {str(k).removeprefix("module."): v for k, v in checkpoint.items()
             if torch.is_tensor(v)}
    model = SRCNN()
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Could not match checkpoint to SRCNN 9-1-5 (64/32 channels). "
            "Expected conv1/conv2/conv3 weights and biases with shapes "
            "[64,1,9,9], [32,64,1,1], [1,32,5,5].\n" + str(exc)
        ) from exc
    model.to(device).eval()
    return model


def decode_gray(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as im:
        return np.asarray(im.convert("L"), dtype=np.float32) / np.float32(255.0)


def load_pair(case: str, data_root: Path | None, archive: Path | None) -> Tuple[np.ndarray, np.ndarray]:
    filename = f"{case}.png"
    if data_root is not None:
        ip, mp = data_root / "images" / filename, data_root / "masks" / filename
        if not ip.is_file() or not mp.is_file():
            raise FileNotFoundError(f"Missing image or mask for {case}: {ip}, {mp}")
        return decode_gray(ip.read_bytes()), decode_gray(mp.read_bytes())
    if archive is None or not archive.is_file():
        raise FileNotFoundError("Dataset ZIP not found; pass --zip or --data-root.")
    with zipfile.ZipFile(archive) as zf:
        ip, mp = f"images/{filename}", f"masks/{filename}"
        try:
            return decode_gray(zf.read(ip)), decode_gray(zf.read(mp))
        except KeyError as exc:
            raise FileNotFoundError(f"Archive is missing {ip} or {mp}") from exc


def bicubic_input(image: np.ndarray) -> np.ndarray:
    """Downsample by 4 then bicubic-upsample to original image dimensions."""
    h, w = image.shape
    low_w, low_h = max(1, w // 4), max(1, h // 4)
    im = Image.fromarray(np.clip(np.round(image * 255), 0, 255).astype(np.uint8), mode="L")
    low = im.resize((low_w, low_h), resample=Image.Resampling.BICUBIC)
    high = low.resize((w, h), resample=Image.Resampling.BICUBIC)
    return np.asarray(high, dtype=np.float32) / np.float32(255.0)


def run_srcnn(model: SRCNN, inp: np.ndarray, device: torch.device) -> np.ndarray:
    x = torch.from_numpy(np.ascontiguousarray(inp[None, None])).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        y = model(x).clamp(0.0, 1.0)[0, 0].cpu().numpy()
    return y.astype(np.float32, copy=False)


def calculate_psnr(reference: np.ndarray, output: np.ndarray) -> float:
    mse = float(np.mean((reference - output) ** 2, dtype=np.float64))
    return math.inf if mse == 0 else float(10.0 * math.log10(1.0 / mse))


def calculate_ssim(reference: np.ndarray, output: np.ndarray) -> float:
    """SSIM with Gaussian sigma=1.5, 7x7 support, K1=.01, K2=.03, range=1."""
    # Truncate at 2 sigma gives exactly radius 3 (window size 7).
    filt = lambda a: gaussian_filter(a, sigma=1.5, mode="reflect", truncate=2.0)
    mu_x, mu_y = filt(reference), filt(output)
    ex2, ey2, exy = filt(reference * reference), filt(output * output), filt(reference * output)
    vx = np.maximum(ex2 - mu_x * mu_x, 0.0)
    vy = np.maximum(ey2 - mu_y * mu_y, 0.0)
    cov = exy - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (vx + vy + c2)
    )
    return float(np.mean(score, dtype=np.float64))


def cnr_values(output: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    lesion = output[mask]
    background = output[~mask]
    if lesion.size == 0 or background.size == 0:
        return math.nan, math.nan
    ml, mb = float(np.mean(lesion)), float(np.mean(background))
    vl = float(np.var(lesion, ddof=1)) if lesion.size > 1 else 0.0
    vb = float(np.var(background, ddof=1)) if background.size > 1 else 0.0
    # Legacy: contrast divided by root-sum-square of ROI standard deviations.
    legacy_denom = math.sqrt(vl + vb)
    legacy = abs(ml - mb) / legacy_denom if legacy_denom > 0 else math.nan
    # Pooled: pooled within-region standard deviation, weighted by ROI sample counts.
    denom_df = lesion.size + background.size - 2
    pooled_var = ((lesion.size - 1) * vl + (background.size - 1) * vb) / denom_df if denom_df > 0 else 0.0
    pooled = abs(ml - mb) / math.sqrt(pooled_var) if pooled_var > 0 else math.nan
    return float(pooled), float(legacy)


def calculate_epi(reference: np.ndarray, output: np.ndarray) -> float:
    """Pearson correlation of Sobel gradient magnitudes over the full image."""
    def gradmag(a):
        gx, gy = sobel(a, axis=1, mode="reflect") / 8.0, sobel(a, axis=0, mode="reflect") / 8.0
        return np.hypot(gx, gy).ravel().astype(np.float64)
    a, b = gradmag(reference), gradmag(output)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def summarize(label: str, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        print(f"{label}: no finite values")
        return
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    print(f"{label} (N={values.size}): Mean={np.mean(values):.6f}, Std={np.std(values, ddof=1):.6f}, "
          f"Median={median:.6f}, IQR={q3-q1:.6f}")


def main() -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=here / "srcnn_x4.pth")
    parser.add_argument("--zip", dest="archive", type=Path, default=here / "Breast image and its masks.zip")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Unzipped dataset directory containing images/ and masks/; takes precedence over --zip.")
    parser.add_argument("--output", type=Path, default=Path("srcnn_per_case_metrics.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if not args.weights.is_file():
        parser.error(f"Weights file not found: {args.weights}")
    device = torch.device(args.device)
    model = load_model(args.weights, device)

    rows = []
    for case in CASES:
        original, mask_gray = load_pair(case, args.data_root, args.archive)
        if original.ndim != 2 or mask_gray.shape != original.shape:
            raise ValueError(f"Image/mask must be aligned grayscale arrays for {case}; got {original.shape}, {mask_gray.shape}")
        mask = mask_gray > 0
        inp = bicubic_input(original)
        output = run_srcnn(model, inp, device)
        if output.shape != original.shape:
            raise RuntimeError(f"SRCNN output size mismatch for {case}: {output.shape} vs {original.shape}")
        area = int(mask.sum())
        if area:
            cnr_pooled, cnr_legacy = cnr_values(output, mask)
            epi = calculate_epi(original, output)
        else:
            cnr_pooled = cnr_legacy = epi = math.nan
        rows.append({
            "case": case, "method": "SRCNN", "scale": 4,
            "height": int(original.shape[0]), "width": int(original.shape[1]),
            "roi_pixels": area, "PSNR_dB": calculate_psnr(original, output),
            "SSIM": calculate_ssim(original, output),
            "CNR_pooled": cnr_pooled, "CNR_legacy": cnr_legacy, "EPI": epi,
        })
        print(f"{case}: PSNR={rows[-1]['PSNR_dB']:.4f} dB, SSIM={rows[-1]['SSIM']:.6f}, "
              f"mask pixels={area}")

    import csv
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["case", "method", "scale", "height", "width", "roi_pixels", "PSNR_dB", "SSIM",
               "CNR_pooled", "CNR_legacy", "EPI"]
    with args.output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote per-case metrics: {args.output.resolve()}")
    print("Summary statistics (sample standard deviation; IQR=Q3-Q1):")
    summarize("PSNR_dB", np.array([r["PSNR_dB"] for r in rows]))
    summarize("SSIM", np.array([r["SSIM"] for r in rows]))
    summarize("CNR_pooled", np.array([r["CNR_pooled"] for r in rows]))
    summarize("CNR_legacy", np.array([r["CNR_legacy"] for r in rows]))
    summarize("EPI", np.array([r["EPI"] for r in rows]))
    expected_valid = 20 - len(EMPTY_MASK_CASES)
    print(f"Expected non-empty mask cases: {expected_valid}; cases expected empty: {', '.join(sorted(EMPTY_MASK_CASES))}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
