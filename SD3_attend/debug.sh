export CUDA_VISIBLE_DEVICES=0
python -m debugpy \
    --listen localhost:15678 \
    --wait-for-client \
    /data1/fanghaipeng/project/sora/tomesd/SD3/test_tome.py \