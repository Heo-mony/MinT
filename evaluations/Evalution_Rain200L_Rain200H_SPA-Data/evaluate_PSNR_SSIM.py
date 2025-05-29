import os
import cv2
import numpy as np
from glob import glob
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2ycbcr


def compute_ssim(img1, img2, K=(0.01, 0.03), window_size=11, L=255):
    # 将图像转换为 YCbCr 并取 Y 通道
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = rgb2ycbcr(img1)[:, :, 0]
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = rgb2ycbcr(img2)[:, :, 0]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (K[0] * L) ** 2
    C2 = (K[1] * L) ** 2

    window = _fspecial_gauss_2d(window_size, 1.5)

    mu1 = gaussian_filter(img1, window_size / 2.576)
    mu2 = gaussian_filter(img2, window_size / 2.576)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 * img1, window_size / 2.576) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, window_size / 2.576) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, window_size / 2.576) - mu1_mu2

    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
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

    return ssim_map.mean()


def compute_psnr(img1, img2):
    # 同样只使用 Y 通道进行评估
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        img1 = rgb2ycbcr(img1)[:, :, 0]
    if len(img2.shape) == 3 and img2.shape[2] == 3:
        img2 = rgb2ycbcr(img2)[:, :, 0]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    rmse = np.sqrt(np.mean((img1 - img2) ** 2))
    if rmse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / rmse)


def _fspecial_gauss_2d(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    if isinstance(size, int):
        size = (size, size)  # 如果是整数，则视为正方形窗口

    m, n = [(ss - 1.) / 2. for ss in size]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


# 主程序逻辑
if __name__ == '__main__':
    datasets = ['DID-Data']  # 可扩展为多个数据集：['Rain200L', 'Rain200H', 'SPA-Data']

    total_psnr = 0
    total_ssim = 0
    num_datasets = len(datasets)

    for dataset in datasets:
        print(f"Processing dataset: {dataset}")

        file_path = f'./results/{dataset}/'
        gt_path = f'./Datasets/{dataset}/test/target/'

        # 获取图像列表（支持 .jpg 和 .png）
        input_files = sorted(glob(os.path.join(file_path, '*.jpg')) + glob(os.path.join(file_path, '*.png')))
        target_files = sorted(glob(os.path.join(gt_path, '*.jpg')) + glob(os.path.join(gt_path, '*.png')))

        if not input_files or not target_files:
            print(f"No image pairs found for dataset {dataset}")
            continue

        assert len(input_files) == len(target_files), "Mismatch between input and ground truth images"

        psnr_sum = 0
        ssim_sum = 0
        count = 0

        for i in range(len(input_files)):
            input_img = cv2.imread(input_files[i])
            gt_img = cv2.imread(target_files[i])

            if input_img is None or gt_img is None:
                print(f"Failed to load image pair: {input_files[i]} or {target_files[i]}")
                continue

            # 计算 PSNR 和 SSIM
            psnr_val = compute_psnr(input_img, gt_img)
            ssim_val = compute_ssim(input_img, gt_img)

            psnr_sum += psnr_val
            ssim_sum += ssim_val
            count += 1

        avg_psnr = psnr_sum / count if count > 0 else 0
        avg_ssim = ssim_sum / count if count > 0 else 0

        print(f"For {dataset} dataset: PSNR = {avg_psnr:.4f}, SSIM = {avg_ssim:.4f}\n")

        total_psnr += avg_psnr
        total_ssim += avg_ssim

    # 输出总平均结果
    overall_avg_psnr = total_psnr / num_datasets if num_datasets > 0 else 0
    overall_avg_ssim = total_ssim / num_datasets if num_datasets > 0 else 0
    print(f"For all datasets: PSNR = {overall_avg_psnr:.4f}, SSIM = {overall_avg_ssim:.4f}")