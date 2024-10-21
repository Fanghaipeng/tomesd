import torch
import torch.distributed as dist
import sys
sys.path.insert(0, '/data1/fanghaipeng/project/sora/DiT')
from models import DiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse

def create_npz_from_sample_folder(sample_dir, num):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}_{num}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def main(args):
    sample_folder_dir = "/data1/fanghaipeng/project/sora/tomesd/DiT/samples/DiT-XL-2-pretrained-size-512-vae-ema-cfg-1.5-seed-0_50k"
    num_fid_samples = 3000
    create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
    sample_folder_dir = "/data1/fanghaipeng/project/sora/tomesd/DiT/samples/DiT-XL-2-pretrained-size-512-vae-ema-cfg-1.5-seed-0_50k"
    num_fid_samples = 6000
    create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
    sample_folder_dir = "/data1/fanghaipeng/project/sora/tomesd/DiT/samples/DiT-XL-2-pretrained-size-512-vae-ema-cfg-1.5-seed-0_50k"
    num_fid_samples = 12500
    create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
    sample_folder_dir = "/data1/fanghaipeng/project/sora/tomesd/DiT/samples/DiT-XL-2-pretrained-size-512-vae-ema-cfg-1.5-seed-0_50k"
    num_fid_samples = 25000
    create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)
    sample_folder_dir = "/data1/fanghaipeng/project/sora/tomesd/DiT/samples/DiT-XL-2-pretrained-size-512-vae-ema-cfg-1.5-seed-0_50k"
    num_fid_samples = 50000
    create_npz_from_sample_folder(sample_folder_dir, num_fid_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)