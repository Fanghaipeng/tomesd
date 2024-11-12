export CUDA_VISIBLE_DEVICES=2

# torchrun \
#     --master_port=36688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
#     --tome-type ToMe --merge-attn --merge-mlp \
#     --batch-size 4 \
#     --ratio-start 1 --ratio-end 1 \
#     --mac-test --test-time 2 \

# torchrun \
#     --master_port=36688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
#     --tome-type ToMe --merge-attn --merge-mlp \
#     --batch-size 4 \
#     --ratio-start 0.5 --ratio-end 0.5 \
#     --mac-test --test-time 5 \

# torchrun \
#     --master_port=36688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
#     --tome-type ToMe --merge-attn --merge-mlp \
#     --batch-size 4 \
#     --ratio-start 0.5 --ratio-end 1 \
#     --mac-test --test-time 5 \

# torchrun \
#     --master_port=36688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
#     --tome-type ToMe_STD --merge-attn --merge-mlp --STD-step 20\
#     --batch-size 4 \
#     --ratio-start 0.5 --ratio-end 0.5 \
#     --mac-test --test-time 5 \
#     --unmerge-steps "0,49" --unmerge-layers "0,4,8,12,16,20,23" \

# torchrun \
#     --master_port=36688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
#     --tome-type ToMe_STD --merge-attn --merge-mlp --STD-step 20\
#     --batch-size 4 \
#     --ratio-start 0.4 --ratio-end 0.4 \
#     --mac-test --test-time 5 \
#     --unmerge-steps "0,49" --unmerge-layers "0,4,8,12,16,20,23" \

# torchrun \
#     --master_port=36688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
#     --tome-type ToMe_STD --merge-attn --merge-mlp --STD-step 20\
#     --batch-size 4 \
#     --ratio-start 0.5 --ratio-end 0.5 \
#     --mac-test --test-time 5 \
#     --metric-times 4 \
    
# torchrun \
#     --master_port=36688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
#     --tome-type ToMe_STD --merge-attn --merge-mlp --STD-step 20\
#     --batch-size 4 \
#     --ratio-start 0.5 --ratio-end 0.5 \
#     --mac-test --test-time 5 \
#     --unmerge-steps "0,49" --unmerge-layers "0,4,8,12,16,20,23" \
#     --metric-times 4 \

# torchrun \
#     --master_port=56688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 50 --guidance-scale 7.0 \
#     --tome-type ToMe_STD --merge-attn --STD-step 20\
#     --batch-size 4 \
#     --ratio-start 0.5 --ratio-end 0.5 \
#     --mac-test --test-time 2 \



# torchrun \
#     --master_port=36688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float16 --seed 0 --height 512 --width 512 --num_inference_steps 50 --guidance-scale 7.0 \
#     --tome-type ToMe --merge-attn --merge-mlp \
#     --batch-size 1 \
#     --ratio-start 1 --ratio-end 1 \
#     --mac-test --test-time 2 \

# torchrun \
#     --master_port=56688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
#     --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
#     --torch-dtype float16 --seed 0 --height 512 --width 512 --num_inference_steps 50 --guidance-scale 7.0 \
#     --tome-type ToMe_STD --merge-attn --STD-step 20\
#     --batch-size 1 \
#     --ratio-start 0.5 --ratio-end 0.5 \
#     --mac-test --test-time 2 \

torchrun \
    --master_port=56688 --nnodes=1 --nproc_per_node=1 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1536 --width 1536 --num_inference_steps 20 --guidance-scale 7.0 \
    --tome-type ToMe_STD\
    --batch-size 1 \
    --ratio-start 1 --ratio-end 1 \
    --mac-test --test-time 1 \

torchrun \
    --master_port=56688 --nnodes=1 --nproc_per_node=1 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/SD3/samples \
    --model-path /data1/fanghaipeng/checkpoints/StableDiffusion/stable-diffusion-3-medium-diffusers \
    --torch-dtype float16 --seed 0 --height 1536 --width 1536 --num_inference_steps 20 --guidance-scale 7.0 \
    --tome-type ToMe_STD --merge-attn --STD-step 20 \
    --batch-size 1 \
    --ratio-start 0.5 --ratio-end 0.5 \
    --mac-test --test-time 1 \