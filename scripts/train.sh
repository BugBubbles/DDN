export CUDA_VISIBLE_DEVICES=0
pyd main.py --base configs/craters-unet-cl.yaml -t \
                --gpus=0,
# --strategy=ddp_find_unused_parameters_true