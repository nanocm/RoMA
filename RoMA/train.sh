torchrun --nproc_per_node=8 --nnodes=1 --master_port 6886 main_pretrain.py  --batch_size 1 --mamba --model arm_base_pz16 --output_dir test/192resolution --norm_pix_loss --blr 1.5e-4 --weight_decay 0.05  --num_workers 12 --enable_flash_attention2 --multi_epochs_dataloader --epochs 800 --warmup_epochs 40

