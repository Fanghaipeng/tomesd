export CUDA_VISIBLE_DEVICES=3
python -m debugpy \
    --listen localhost:15678 \
    --wait-for-client \
    /data1/fanghaipeng/project/sora/tomesd/demo.py \
