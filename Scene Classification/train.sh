CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --standalone main_linprobe.py \
--dataset 'aid' --model 'arm_base_pz16' --input_size 224 --postfix 'sota' \
--batch_size 128 --epochs 200 --warmup_epochs 5 --accum_iter 4 \
--blr 1.5e-3  --weight_decay 0.05 --split 55 --tag 0 --exp_num 1 --ema_decay 0.99992  --dist_eval --layer_decay 0.75 \
--drop_path 0.0 \
--finetune /root/iccv-base-100w-440e.pth



