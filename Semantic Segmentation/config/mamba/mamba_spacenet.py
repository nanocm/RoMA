_base_ = [
    '../_base_/models/upernet_mamba_base.py', #'../_base_/datasets/spacenet.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

dataset_type = 'SPACENETV1Dataset'
data_root = '/root/spacenet'
img_norm_cfg = dict(
    mean=[121.826, 106.52838, 78.372116], std=[56.717068, 44.517075, 40.451515], to_rgb=True)
crop_size = (384, 384)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(384, 384), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(384,384),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]


data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/images',
        ann_dir='train/labels',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/labels',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/images',
        ann_dir='val/labels',
        pipeline=test_pipeline))
        
# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05,
                 constructor='LayerDecayOptimizerConstructor', 
                 paramwise_cfg=dict(
                        num_layers=12, 
                        layer_decay_rate=0.9,
                        layer_sep = '.',
                            )
                            )

lr_config = dict(_delete_=True, policy='CosineAnnealing',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 min_lr=0.0, by_epoch=False)
 
optimizer_config = dict(grad_clip=None)

model = dict(
    pretrained='/root/final-base-400w-400epoch.pth',
    backbone=dict(
        type='arm_base_pz16',
        img_size=384,
        drop_rate=0.,
        drop_path_rate=0.1,
        ),
    decode_head=dict(
        num_classes=2,
        ignore_index=255
    ),
    auxiliary_head=dict(
        num_classes=2,
        ignore_index=255
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(256,256), crop_size=(384,384))
    )
