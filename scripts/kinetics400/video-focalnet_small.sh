python -m torch.distributed.launch --nproc_per_node 8  main.py \
--cfg configs/kinetics400/video-focalnet_small.yaml