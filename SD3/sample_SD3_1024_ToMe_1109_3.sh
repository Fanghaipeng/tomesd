export CUDA_VISIBLE_DEVICES=4,5

torchrun \
    --master_port=58888 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe_STD --merge-attn --STD-step 20 \
    --batch-size 8 \
    --ratio-start 1 --ratio-end 1 \

torchrun \
    --master_port=58888 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe --merge-attn --STD-step 20 \
    --batch-size 8 \
    --ratio-start 0.5 --ratio-end 0.5 \

torchrun \
    --master_port=58888 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe --merge-attn --STD-step 20 \
    --batch-size 8 \
    --ratio-start 0.7 --ratio-end 0.7 \

torchrun \
    --master_port=58888 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe --merge-attn --STD-step 20 \
    --batch-size 8 \
    --ratio-start 0.6 --ratio-end 0.6 \

torchrun \
    --master_port=58888 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe --merge-attn --STD-step 20 \
    --batch-size 8 \
    --ratio-start 0.8 --ratio-end 0.8 \

torchrun \
    --master_port=58888 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe --merge-attn --STD-step 20 \
    --batch-size 8 \
    --ratio-start 0.9 --ratio-end 0.9 \

torchrun \
    --master_port=58888 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe --merge-attn --STD-step 20 \
    --batch-size 8 \
    --ratio-start 0.6 --ratio-end 1 \

torchrun \
    --master_port=58888 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe --merge-attn --STD-step 20 \
    --batch-size 8 \
    --ratio-start 0.7 --ratio-end 1 \

torchrun \
    --master_port=58888 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe --merge-attn --STD-step 20 \
    --batch-size 8 \
    --ratio-start 0.8 --ratio-end 1 \

torchrun \
    --master_port=58888 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe --merge-attn --STD-step 20 \
    --batch-size 8 \
    --ratio-start 0.9 --ratio-end 1 \


