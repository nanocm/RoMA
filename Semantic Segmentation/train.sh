torchrun --standalone --nproc_per_node=4 --master_port=30000 tools/train.py \
configs/mamba/mamba_spacenet.py --launcher 'pytorch'  --cfg-options 'find_unused_parameters'=True --no-validate