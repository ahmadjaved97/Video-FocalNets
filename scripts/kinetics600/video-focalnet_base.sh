python -m torch.distributed.launch --nproc_per_node 4 main.py \
--cfg configs/kinetics600/video-focalnet_base.yaml