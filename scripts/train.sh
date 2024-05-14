export CUDA_VISIBLE_DEVICES=3,4
python main.py --base configs/craters-unet-cl.yaml -t --gpus=0,1, --strategy=ddp_find_unused_parameters_true