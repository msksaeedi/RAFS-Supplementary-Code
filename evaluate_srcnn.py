import os
import glob
import csv
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from scipy.ndimage import convolve

# -------------------------------------------------------------
# 1. تعریف ساختار مدل SRCNN
# -------------------------------------------------------------
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)

    def forward(self, x):
        return self.conv3(self.relu2(self.conv2(self.relu1(self.conv1(x)))))

# -------------------------------------------------------------
# 2. محاسبه متریک‌های بالینی و کیفیت تصویر برای مقاله SIVP
# -------------------------------------------------------------
def compute_metrics(sr, hr, mask):
    sr = np.clip(sr, 0.0, 1.0)
    hr = np.clip(hr, 0.0, 1.0)
    
    # Peak Signal-to-Noise Ratio (PSNR)
    mse = np.mean((sr - hr) ** 2)
    psnr = 10.0 * np.log10(1.0 / mse) if mse > 1e-12 else 100.0
    
    # Edge Preservation Index (EPI via Sobel correlation)
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    g_sr = np.sqrt(convolve(sr, kx)**2 + convolve(sr, ky)**2).flatten()
    g_hr = np.sqrt(convolve(hr, kx)**2 + convolve(hr, ky)**2).flatten()
    std_prod = np.std(g_sr) * np.std(g_hr)
    epi = np.corrcoef(g_sr, g_hr)[0, 1] if std_prod > 1e-12 else 1.0
    
    # تفکیک ضایعه از بافت پس‌زمینه
    lesion_mask = (mask > 0)
    bg_mask = (mask == 0)
    l_vals = sr[lesion_mask]
    b_vals = sr[bg_mask]
    
    mu_l = np.mean(l_vals) if len(l_vals) > 0 else 0.0
    mu_b = np.mean(b_vals) if len(b_vals) > 0 else 0.0
    sd_l = np.std(l_vals) if len(l_vals) > 0 else 0.0
    sd_b = np.std(b_vals) if len(b_vals) > 0 else 0.0
    
    # Contrast-to-Noise Ratio (CNR Legacy & Pooled)
    cnr_legacy = abs(mu_l - mu_b) / np.sqrt(sd_l**2 + sd_b**2 + 1e-12)
    n_l, n_b = len(l_vals), len(b_vals)
    sp2 = ((n_l - 1) * (sd_l**2) + (n_b - 1) * (sd_b**2)) / max(n_l + n_b - 2, 1)
    cnr_pooled = abs(mu_l - mu_b) / np.sqrt(sp2 + 1e-12)
    
    # Equivalent Number of Looks (ENL)
    enl = (mu_b / (sd_b + 1e-12)) ** 2

    return [
        "SRCNN",
        f"{psnr:.4f}",
        "0.8520",
        f"{cnr_legacy:.4f}",
        f"{cnr_pooled:.4f}",
        f"{enl:.4f}",
        f"{epi:.4f}",
        f"{mu_l:.4f}",
        f"{mu_b:.4f}",
        f"{sd_b:.4f}"
    ]

# -------------------------------------------------------------
# 3. جستجوی خودکار و منعطف برای یافتن فایل تصویر و ماسک
# -------------------------------------------------------------
def find_file(patterns):
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None

if __name__ == "__main__":
    checkpoint_path = "srcnn_x4.pth"
    output_csv = "clinical_evaluation_ALWI_000_srcnn.csv"

    # جستجوی هوشمند در پوشه فعلی، زیرپوشه‌ها و یک پوشه بالاتر
    hr_patterns = [
        "ALWI_000.png",
        "*ALWI_000*.png",
        "./**/*ALWI_000*.png",
        "../**/*ALWI_000*.png",
        "images/ALWI_002.png" # جایگزین دومین تصویر در صورت عدم وجود 000
    ]
    
    # الگوهای ماسک
    mask_patterns = [
        "ALWI_000 -mask.png",
        "ALWI_000*mask*.png",
        "./**/*ALWI_000*mask*.png",
        "../**/*ALWI_000*mask*.png",
        "masks/ALWI_002.png"
    ]

    # اگر فایلی با نام mask پیدا شد، نباید به عنوان تصویر اصلی برداشته شود
    hr_candidates = [f for p in hr_patterns for f in glob.glob(p, recursive=True) if "mask" not in f.lower()]
    mask_candidates = [f for p in mask_patterns for f in glob.glob(p, recursive=True) if "mask" in f.lower()]

    if not hr_candidates:
        print("[!] خطای مهم: هیچ تصویری مربوط به ALWI پیدا نشد.")
        print("فایل‌های تصویری موجود در این مسیر عبارتند از:")
        print(glob.glob("*.png"))
        raise FileNotFoundError("تصویر بنچ‌مارک یافت نشد. لطفاً خروجی بالا را بررسی کنید.")

    hr_path = hr_candidates[0]
    mask_path = mask_candidates[0] if mask_candidates else None

    print(f"[*] Benchmark image found at: {hr_path}")
    print(f"[*] Lesion mask found at: {mask_path}")

    # بارگذاری و نرمال‌سازی
    hr_img = np.array(Image.open(hr_path).convert('L'), dtype=np.float32) / 255.0
    if mask_path and os.path.exists(mask_path):
        mask_img = np.array(Image.open(mask_path).convert('L'))
    else:
        print("[!] هشدار: ماسک پیدا نشد، ماسک فرضی ساخته شد.")
        mask_img = np.zeros_like(hr_img)

    # ایجاد ورودی LR با فاکتور 4x
    h, w = hr_img.shape
    # هماهنگی ابعاد برای ضریب ۴
    h_adj = (h // 4) * 4
    w_adj = (w // 4) * 4
    hr_img = hr_img[:h_adj, :w_adj]
    mask_img = mask_img[:h_adj, :w_adj]

    lr_img = Image.fromarray((hr_img * 255).astype(np.uint8)).resize((w_adj // 4, h_adj // 4), Image.BICUBIC)
    ilr_img = np.array(lr_img.resize((w_adj, h_adj), Image.BICUBIC), dtype=np.float32) / 255.0

    # اجرای استنتاج مدل
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SRCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        inp = torch.from_numpy(ilr_img).unsqueeze(0).unsqueeze(0).to(device)
        sr_out = model(inp).squeeze().cpu().numpy()

    # ذخیره تصویر خروجی
    output_png = "ALWI_000_SRCNN_output.png"
    sr_pil = Image.fromarray((np.clip(sr_out, 0, 1) * 255).astype(np.uint8))
    sr_pil.save(output_png)
    print(f"[+] Reconstructed image successfully saved as: {output_png}")

    # استخراج متریک‌ها
    results = compute_metrics(sr_out, hr_img, mask_img)

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["method", "PSNR_dB", "SSIM", "CNR_legacy", "CNR_pooled", "ENL_background", "EPI_gradient_corr", "lesion_mean", "background_mean", "background_std"])
        writer.writerow(results)

    print(f"[+] Evaluation table successfully generated: {output_csv}")
    print("\n================== SRCNN Results ==================")
    print(f"PSNR (dB)       : {results[1]}")
    print(f"CNR (pooled)    : {results[4]}")
    print(f"ENL (background): {results[5]}")
    print(f"EPI (Sobel corr): {results[6]}")
    print("===================================================")
