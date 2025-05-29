import os
import numpy as np
from PIL import Image
from skimage.color import rgb2ycbcr
from scipy.ndimage import convolve, gaussian_filter
from skimage.transform import resize

# PSNR 计算函数
def psnr(f1, f2):
    if np.max(f1) > 2.0 or np.max(f2) > 2.0:
        k = 8
    else:
        k = 1

    fmax = (2 ** k) - 1
    a = fmax ** 2

    f1 = f1.astype(np.float64)
    f2 = f2.astype(np.float64)

    mse = np.mean((f1 - f2) ** 2)

    if mse == 0:
        return float('inf')

    psnr_val = 10 * np.log10(a / mse)
    return psnr_val

# SSIM 计算函数
def ssim(img1, img2, K=(0.01, 0.03), window=None, L=255):
    if len(img1.shape) > 2:
        img1 = np.mean(img1, axis=2)
    if len(img2.shape) > 2:
        img2 = np.mean(img2, axis=2)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    M, N = img1.shape

    if window is None:
        window = _fspecial_gauss_2d((11, 11), sigma=1.5)
    else:
        H, W = window.shape
        if H * W < 4 or H > M or W > N:
            return -np.inf, -np.inf

    if len(K) != 2 or K[0] < 0 or K[1] < 0:
        return -np.inf, -np.inf

    # 自动下采样
    f = max(1, round(min(M, N) / 256))
    if f > 1:
        lpf = np.ones((f, f)) / (f * f)
        img1 = convolve(img1, lpf, mode='reflect')
        img2 = convolve(img2, lpf, mode='reflect')
        img1 = resize(img1, (M // f, N // f), anti_aliasing=False, preserve_range=True)
        img2 = resize(img2, (M // f, N // f), anti_aliasing=False, preserve_range=True)

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2
    window /= np.sum(window)

    mu1 = convolve(img1, window, mode='reflect')
    mu2 = convolve(img2, window, mode='reflect')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve(img1 * img1, window, mode='reflect') - mu1_sq
    sigma2_sq = convolve(img2 * img2, window, mode='reflect') - mu2_sq
    sigma12 = convolve(img1 * img2, window, mode='reflect') - mu1_mu2

    if C1 > 0 and C2 > 0:
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim_map = numerator / denominator
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = np.ones_like(mu1)
        index = (denominator1 * denominator2) > 0
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])

        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    mssim = np.mean(ssim_map)

    return mssim, ssim_map

def _fspecial_gauss_2d(size, sigma):
    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# 主评估函数
def evaluate_dataset(num_samples=1200, result_dir='./results'):
    total_psnr = 0
    total_ssim = 0

    for i in range(1, num_samples + 1):
        # 构造文件路径
        gt_path = os.path.join(result_dir, f'{i}_GT.png')
        dr_path = os.path.join(result_dir, f'{i}_DR.png')

        # 读取图像
        try:
            img_gt = np.array(Image.open(gt_path).convert('RGB')) / 255.0
            img_dr = np.array(Image.open(dr_path).convert('RGB')) / 255.0
        except FileNotFoundError as e:
            print(f"Missing file: {e}")
            continue

        # 转换到 YCbCr 并提取 Y 通道
        y_gt = rgb2ycbcr(img_gt)[:, :, 0]
        y_dr = rgb2ycbcr(img_dr)[:, :, 0]

        # 计算 PSNR 和 SSIM
        p = psnr(y_dr * 255, y_gt * 255)
        s = ssim(y_dr * 255, y_gt * 255)[0]  # 只取 mssim

        total_psnr += p
        total_ssim += s

        print(f"Image {i}: PSNR={p:.4f}, SSIM={s:.4f}")

    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples

    print(f"\n{'='*40}")
    print(f"Average PSNR = {avg_psnr:.4f} dB")
    print(f"Average SSIM = {avg_ssim:.4f}")
    print(f"{'='*40}")

    return avg_psnr, avg_ssim

# 执行评估
if __name__ == "__main__":
    # 对于 DID-Data（1200 张）
    evaluate_dataset(num_samples=1200, result_dir='./results')

    # 若要对 DDN-Data（1400 张）进行评估，取消注释以下行：
    # evaluate_dataset(num_samples=1400, result_dir='./results')