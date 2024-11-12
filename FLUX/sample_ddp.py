import json
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import os

from tqdm import tqdm
from tome import apply_patch_ToMe
# from counter import show_timing_info
# from tome_STD import apply_patch_ToMe_STD
# from diffusers import StableDiffusion3Pipeline
from pipeline_flux import FluxPipeline
from counter import MacTracker, TimeTracker
# from deepspeed.profiling.flops_profiler import FlopsProfiler
import time

import debugpy

## 启动debugpy，监听指定端口
# debugpy.listen(('0.0.0.0', 15678))
# print("Waiting for debugger to attach...")
# debugpy.wait_for_client()
# print("Debugger attached!")

def load_captions(file_path):
    """
    Reads the 'annotations' section of a specified JSON file, extracts the image_id and caption from each annotation,
    and returns a list containing this data, ensuring that each image_id is unique and has the longest caption.

    :param file_path: Path to the JSON file
    :return: A list where each element is a dictionary containing 'image_id' and the longest 'caption' for that image_id
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    annotations = data['annotations']
    # Dictionary to store the longest caption for each image_id
    longest_captions = {}

    # Iterate through each annotation
    for item in annotations:
        image_id = item['image_id']
        caption = item['caption']
        # If the image_id is already in the dictionary and the current caption is longer, update it
        if image_id in longest_captions:
            if len(caption) > len(longest_captions[image_id]['caption']):
                longest_captions[image_id] = {'image_id': image_id, 'caption': caption}
        else:
            # Otherwise, add the image_id and caption to the dictionary
            longest_captions[image_id] = {'image_id': image_id, 'caption': caption}

    # Extract values from the dictionary to form the final list
    image_captions = list(longest_captions.values())

    return image_captions

def main(args):
    """
    Run sampling.
    """
    # Initialize the process group
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    file_path = args.caption_path
    captions_list = load_captions(file_path)
    # 保存为 JSON 文件
    with open('/data/fanghaipeng/datasets/COCO2017/longest_captions.json', 'w', encoding='utf-8') as file:
        json.dump(captions_list, file, ensure_ascii=False, indent=4)

    if rank == 0:
        print(f"Loaded unduplicated {len(captions_list)} captions.")

    # Load the pipeline
    if args.torch_dtype == "float32":
        pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.float32)
    elif args.torch_dtype == "float16":
        pipe = FluxPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipe = pipe.to(device)

    # Ensure all processes have loaded the model before proceeding
    dist.barrier()

    # Construct output path
    output_path = os.path.join(
        args.output_path, # {pipe.__class__.__name__}-
        f"{args.height}-{args.width}-{args.num_inference_steps}-{args.guidance_scale}-{args.torch_dtype}-" \
        f"{args.tome_type}-prune{args.prune_replace}-x:{args.merge_x}-sa:{args.merge_attn}-ca:{args.merge_crossattn}-mlp:{args.merge_mlp}-"
        f"step{args.STD_step}-s{args.ratio_start}-e{args.ratio_end}-metric{args.metric_times}-unstep{args.unmerge_steps}-unlayer{args.unmerge_layers}"  
    )

    # Create the output directory if it doesn't exist
    if rank == 0:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
    # Wait until the directory is created
    dist.barrier()

    batch_size = args.batch_size
    num_samples = len(captions_list)

    # Split the data among ranks
    indices = list(range(num_samples))
    indices_per_rank = indices[rank::world_size]

    # Build the process's own captions_list
    local_captions_list = [captions_list[i] for i in indices_per_rank]

    generator = torch.Generator(device=device).manual_seed(args.seed)

    if args.tome_type is None or args.tome_type == "default":
        pipe = pipe
    else:
        unmerge_steps = list(map(int, args.unmerge_steps.split(','))) if args.unmerge_steps is not None else []
        unmerge_layers = list(map(int, args.unmerge_layers.split(','))) if args.unmerge_layers is not None else []
        apply_patch_ToMe(pipe, ratio=args.ratio, ratio_start=args.ratio_start, ratio_end=args.ratio_end, sx=2, sy=2, \
            save_dir=output_path, tome_type=args.tome_type, prune_replace=args.prune_replace, STD_step=args.STD_step, merge_x=args.merge_x, \
            merge_attn=args.merge_attn, merge_crossattn=args.merge_crossattn, merge_mlp=args.merge_mlp, \
            trace_source=args.trace_source, speed_test=args.speed_test, mac_test=args.mac_test, \
            unmerge_steps=unmerge_steps, unmerge_layers=unmerge_layers, metric_times=args.metric_times)

    if args.speed_test:
        test_time = args.test_time
        start_time = time.time()

        for i in tqdm(range(0, test_time), desc=f"Rank {rank}"):
            batch_captions = local_captions_list[i: i + batch_size]
            prompt_list = [item['caption'] for item in batch_captions]
            id_list = [item['image_id'] for item in batch_captions]
            images = pipe(
                prompt=prompt_list,
                generator=generator,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                speed_test=True,
            ).images
        image_time = time.time() - start_time
        print(f"Total image for {test_time} batch time spent: {image_time} seconds")
        timetracker = TimeTracker.get_instance()
        # timetracker.show_step(test_time * args.batch_size)
        # timetracker.show_layer(test_time * args.batch_size)
        timetracker.show_category(test_time * args.batch_size)

        return

    if args.mac_test:
        test_time = args.test_time
        for i in tqdm(range(0, test_time), desc=f"Rank {rank}"):
            batch_captions = local_captions_list[i: i + batch_size]
            prompt_list = [item['caption'] for item in batch_captions]
            id_list = [item['image_id'] for item in batch_captions]
            images = pipe(
                prompt=prompt_list,
                generator=generator,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                mac_test=True,
            ).images
        mactracker = MacTracker.get_instance()
        mactracker.show_step(test_time * args.batch_size)
        mactracker.show_avg(test_time * args.batch_size)
        return

    for i in tqdm(range(0, len(local_captions_list), batch_size), desc=f"Rank {rank}"):
        batch_captions = local_captions_list[i: i + batch_size]
        prompt_list = [item['caption'] for item in batch_captions]
        id_list = [item['image_id'] for item in batch_captions]
        
        images = pipe(
            prompt=prompt_list,
            generator=generator,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        ).images
        for j, image in enumerate(images):
            image_id = id_list[j]
            image_id = str(image_id).zfill(12)
            image.save(os.path.join(output_path, f"{image_id}.jpg"))
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--caption-path", type=str, default="/data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json")
    parser.add_argument("--output-path", type=str, default="/data1/fanghaipeng/project/sora/tomesd/SD3/samples")
    parser.add_argument("--model-path", type=str, default="/data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers")
    parser.add_argument("--torch-dtype", type=str, default="float32")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-time", type=int, default=4)
    parser.add_argument("--tome-type", type=str, choices=["default", "ToMe", "ToMe_STD"], default="default")
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--ratio-start", type=float, default=1.0)
    parser.add_argument("--ratio-end", type=float, default=1.0)
    parser.add_argument("--save-step", type=bool, default=False)
    parser.add_argument("--speed-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--mac-test", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prune-replace", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trace-source", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--merge-x", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--merge-attn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--merge-crossattn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--merge-mlp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--STD-step", type=int, default=0)
    parser.add_argument("--unmerge-steps", type=str, help='Specific steps as a comma-separated string')
    parser.add_argument("--unmerge-layers", type=str, help='Specific steps as a comma-separated string')
    parser.add_argument("--metric-times", type=int, default=1)
    args = parser.parse_args()
    main(args)