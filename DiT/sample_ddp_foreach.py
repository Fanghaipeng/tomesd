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
from tome import apply_patch_tome, show_timing_info

import debugpy

# 启动debugpy，监听指定端口
debugpy.listen(('0.0.0.0', 15678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached!")

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
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{args.image_size}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}-{args.tome_type}-prune{args.prune_replace}-{args.ratio}-{args.ratio_start}-{args.ratio_end}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    # ToMe
    if args.tome_type == "default":
        pipe = pipe
    elif args.tome_type == "ToMe":
        apply_patch_tome(model, ratio=args.ratio, ratio_start=args.ratio_start, ratio_end=args.ratio_end, sx=2, sy=2, \
                         save_dir=os.path.join(sample_folder_dir,'debug'), prune_replace=args.prune_replace)
    # elif args.tome_type == "ToMe_STD":
    #     apply_patch_ToMe_STD(pipe, ratio=args.ratio, ratio_start=args.ratio_start, ratio_end=args.ratio_end, sx=4, sy=4, save_dir=output_path)

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    each_class_num = args.num_fid_samples // 1000
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    start_idx = rank * samples_needed_this_gpu
    end_idx = start_idx + samples_needed_this_gpu

    # Generate labels: [0, 0, 1, 1, ..., 999, 999]

    all_labels = torch.tensor([min(i // each_class_num, 999) for i in range(total_samples)], device=device)
    
    labels_this_gpu = all_labels[start_idx:end_idx]

    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    for i in pbar:
        batch_start = i * args.per_proc_batch_size
        batch_end = min(batch_start + args.per_proc_batch_size, samples_needed_this_gpu)
        batch_size = batch_end - batch_start

        # Sample inputs:
        z = torch.randn(batch_size, model.in_channels, latent_size, latent_size, device=device)
        y = labels_this_gpu[batch_start:batch_end]

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * batch_size, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        samples = diffusion.p_sample_loop(
            sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
        )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for idx_in_batch, sample in enumerate(samples):
            index = start_idx + batch_start + idx_in_batch
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")
    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=8)
    # Set default num-fid-samples to 2000
    parser.add_argument("--num-fid-samples", type=int, default=2000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    # Set ToMe
    parser.add_argument("--tome-type", type=str, choices=["default", "ToMe", "ToMe_STD"], default="default")
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--ratio-start", type=float, default=0.5)
    parser.add_argument("--ratio-end", type=float, default=0.5)
    parser.add_argument("--prune-replace", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    main(args)