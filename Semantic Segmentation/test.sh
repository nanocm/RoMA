torchrun --standalone --nproc_per_node=4 tools/test.py configs/mamba/mamba_spacenet.py \
work_dirs/mamba_spacenet/latest.pth \
--eval 'mIoU' --eval-options imgfile_prefix="work_dirs/display/mamba_spacenet/result" \
--show-dir work_dirs/display/mamba_spacenet/rgb --launcher pytorch 