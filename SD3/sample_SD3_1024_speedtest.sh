CUDA_VISIBLE_DEVICES=5 torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=1 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 \
    --height 1024 --width 1024 \
    --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type ToMe \
    --batch-size 4 \
    --ratio 0.5 \
    --ratio-start 0.5 \
    --ratio-end 0.5 \
    --speed-test True

# CUDA_VISIBLE_DEVICES=5 torchrun \
#     --master_port=26688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float16 --seed 0 \
#     --height 1024 --width 1024 \
#     --num_inference_steps 50 --guidance-scale 7.0 \
#     --tome-type tome \
#     --batch-size 4 \
#     --ratio 0.5 \
#     --ratio-start 1 \
#     --ratio-end 1 \
#     --speed-test True