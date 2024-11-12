export CUDA_VISIBLE_DEVICES=1,2,3

torchrun \
    --master_port=51888 --nnodes=1 --nproc_per_node=3 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe_STD --merge-attn --merge-mlp \
    --batch-size 8 \
    --ratio-start 0.5 --ratio-end 0.5 \
    
torchrun \
    --master_port=51888 --nnodes=1 --nproc_per_node=3 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe_STD --merge-attn --merge-mlp \
    --batch-size 8 \
    --ratio-start 0.5 --ratio-end 0.5 \
    --metric-times 4 \
    
torchrun \
    --master_port=51888 --nnodes=1 --nproc_per_node=3 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe_STD --merge-attn --merge-mlp \
    --batch-size 8 \
    --ratio-start 0.5 --ratio-end 0.5 \
    --metric-times 4 \
    --unmerge-steps "0,49" --unmerge-layers "0,4,8,12,16,20,23" \



