export CUDA_VISIBLE_DEVICES=3,5
python main.py --base configs/craters-unet-cl.yaml -t \
               --resume /home/xdy_cbf/CODE_cbf/Destrip/logs/2024-05-14T21-23-00_craters-unet-cl/checkpoints/last.ckpt \
                --gpus=0,1, \
                --strategy=ddp_find_unused_parameters_true