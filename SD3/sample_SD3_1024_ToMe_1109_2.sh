export CUDA_VISIBLE_DEVICES=6,7

torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 10 --guidance-scale 7.0 \
    --tome-type ToMe_STD --merge-attn --STD-step 4 \
    --batch-size 8 \
    --ratio-start 0.8 --ratio-end 1 \

torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 15 --guidance-scale 7.0 \
    --tome-type ToMe_STD --merge-attn --STD-step 6 \
    --batch-size 8 \
    --ratio-start 0.8 --ratio-end 1 \

torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 20 --guidance-scale 7.0 \
    --tome-type ToMe_STD --merge-attn --STD-step 8 \
    --batch-size 8 \
    --ratio-start 0.8 --ratio-end 1 \

torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 28 --guidance-scale 7.0 \
    --tome-type ToMe_STD --merge-attn --STD-step 11 \
    --batch-size 8 \
    --ratio-start 0.8 --ratio-end 1 \

torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 40 --guidance-scale 7.0 \
    --tome-type ToMe_STD --merge-attn --STD-step 16 \
    --batch-size 8 \
    --ratio-start 0.8 --ratio-end 1 \
