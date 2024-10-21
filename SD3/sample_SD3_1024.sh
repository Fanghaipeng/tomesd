CUDA_VISIBLE_DEVICES=4,5 torchrun \
    --master_port=26688 --nnodes=1 --nproc_per_node=2 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 \
    --height 1024 --width 1024 \
    --num_inference_steps 50 --guidance-scale 7.0 \
    --tome-type default \
    --batch-size 8 \

# CUDA_VISIBLE_DEVICES=4,5 torchrun \
#     --master_port=26688 --nnodes=1 --nproc_per_node=2 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float32 --seed 0 \
#     --height 1024 --width 1024 \
#     --num_inference_steps 50 --guidance-scale 7.0 \
#     --batch-size 4 \