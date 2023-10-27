import pandas as pd
import os

# Read in the data
used_dataset = "EB_DAVIS_test"
root_path = f"metrics/{used_dataset}/"

total_psnr = 0
total_ssim = 0
total_fid = 0
total_lpips = 0

len = len(os.listdir(root_path))

# loop through all the folders
for folder in os.listdir(root_path):
    df = pd.read_csv(root_path + folder + "/metrics.csv")

    psnr, ssim, fid, lpips = df.iloc[:,1:].mean()

    total_psnr += psnr
    total_ssim += ssim
    total_fid += fid
    total_lpips += lpips

total_fid /= len
total_lpips /= len
total_psnr /= len
total_ssim /= len

print(f"PSNR: {total_psnr}")
print(f"SSIM: {total_ssim}")
print(f"FID: {total_fid}")
print(f"LPIPS: {total_lpips}")
