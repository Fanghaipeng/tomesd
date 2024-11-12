export CUDA_VISIBLE_DEVICES=4

# torchrun \
#     --master_port=36688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/PixArt/samples \
#     --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 20 --guidance-scale 4.5 \
#     --batch-size 1 \
#     --tome-type default\

# torchrun \
#     --master_port=36688 --nnodes=1 --nproc_per_node=1 \
#     sample_ddp.py \
#     --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
#     --output-path /data1/fanghaipeng/project/sora/tomesd/PixArt/samples \
#     --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 20 --guidance-scale 4.5 \
#     --batch-size 1 \
#     --tome-type ToMe --merge-attn --merge-crossattn --merge-mlp \
#     --ratio-start 1 --ratio-end 1 \

torchrun \
    --master_port=36688 --nnodes=1 --nproc_per_node=1 \
    sample_ddp.py \
    --caption-path /data/fanghaipeng/datasets/COCO2017/annotations/captions_val2017.json \
    --output-path /data1/fanghaipeng/project/sora/tomesd/PixArt/samples \
    --torch-dtype float16 --seed 0 --height 1024 --width 1024 --num_inference_steps 20 --guidance-scale 4.5 \
    --batch-size 1 \
    --tome-type ToMe --merge-attn \
    --ratio-start 0.8 --ratio-end 0.8 \
    # --tome-type ToMe --merge-attn --merge-crossattn --merge-mlp \
    # --ratio-start 0.5 --ratio-end 0.5 \


