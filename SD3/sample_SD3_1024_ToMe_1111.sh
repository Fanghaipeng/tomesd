export CUDA_VISIBLE_DEVICES=6,7

torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 9.0 \
    --tome-type ToMe \
    --batch-size 8 \
    --ratio-start 1 --ratio-end 1 \

torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 6.0 \
    --tome-type ToMe \
    --batch-size 8 \
    --ratio-start 1 --ratio-end 1 \


torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 4.0 \
    --tome-type ToMe \
    --batch-size 8 \
    --ratio-start 1 --ratio-end 1 \


torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 3.0 \
    --tome-type ToMe \
    --batch-size 8 \
    --ratio-start 1 --ratio-end 1 \


torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 2.0 \
    --tome-type ToMe \
    --batch-size 8 \
    --ratio-start 1 --ratio-end 1 \

torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 1.5 \
    --tome-type ToMe \
    --batch-size 8 \
    --ratio-start 1 --ratio-end 1 \

torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 1.0 \
    --tome-type ToMe \
    --batch-size 8 \
    --ratio-start 1 --ratio-end 1 \