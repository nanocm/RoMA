# https://github.com/ViTAE-Transformer/MTP/blob/main/RS_Tasks_Finetune/Change_Detection/configs/mtp/oscd/rvsa-b-unet-96-mae-mtp_oscd_rgb.py
############################### default runtime #################################

default_scope = "opencd"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
vis_backends = [dict(type="CDLocalVisBackend")]
visualizer = dict(
    type="CDLocalVisualizer",
    vis_backends=vis_backends,
    name="visualizer",
    alpha=1.0,
)
log_processor = dict(by_epoch=False)
log_level = "INFO"
load_from = None
resume = False

############################### dataset #################################

# dataset settings
dataset_type = "OSCD_RGB_CD_Dataset"
# data_root = '/work/share/achk2o1zg1/diwang22/dataset/OSCD/oscd_rgb_patch_dataset'
data_root = "/home/chenmingshuo/projects/Mamba/datasets/oscd_rgb_patch_dataset"


size = 768
BS = 4
interval = 1
crop_size = (size, size)
train_pipeline = [
    dict(type="MultiImgLoadImageFromFile"),
    dict(type="MultiImgResize", scale=crop_size, keep_ratio=True),
    dict(type="MultiImgLoadAnnotations"),
    dict(
        type="MultiImgRandomRotFlip",
        rotate_prob=0.5,
        flip_prob=0.5,
        degree=(-20, 20),
    ),
    dict(type="MultiImgRandomCrop", crop_size=crop_size, cat_max_ratio=0.75),
    dict(type="MultiImgExchangeTime", prob=0.5),
    dict(
        type="MultiImgPhotoMetricDistortion",
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10,
    ),
    dict(type="MultiImgPackSegInputs"),
]

val_pipeline = [
    dict(type="MultiImgLoadImageFromFile"),
    dict(type="MultiImgResize", scale=crop_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type="MultiImgLoadAnnotations"),
    dict(type="MultiImgPackSegInputs"),
]
# test_pipeline = [
#     dict(type='MultiImgLoadImageFromFile'),
#     dict(type='MultiImgLoadAnnotations'),
#     dict(type='MultiImgPackSegInputs')
# ]

train_dataloader = dict(
    batch_size=BS,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="train/A",
            img_path_to="train/B",
            seg_map_path="train/label",
        ),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="test/A",
            img_path_to="test/B",
            seg_map_path="test/label",
        ),
        pipeline=val_pipeline,
    ),
)
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path_from="test/A",
            img_path_to="test/B",
            seg_map_path="test/label",
        ),
        pipeline=val_pipeline,
    ),
)

val_evaluator = dict(type="mmseg.IoUMetric", iou_metrics=["mFscore", "mIoU"])
test_evaluator = val_evaluator

############################### running schedule #################################


# optimizer
# optim_wrapper = dict(
#     optimizer=dict(
#         type="AdamW", lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05
#     ),
#     constructor="LayerDecayOptimizerConstructor_ViT",
#     paramwise_cfg=dict(
#         num_layers=12,
#         layer_decay_rate=0.9,
#     ),
# )
optim_wrapper = dict(
    optimizer=dict(
        type="AdamW",
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.05,
    ),
    constructor="LayerDecayOptimizerConstructor",
    paramwise_cfg=dict(
        num_layers=12,
        layer_decay_rate=0.75,
        layer_sep=".",
        custom_keys={
            "bias": dict(decay_multi=0.0),
            "pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
            "rel_pos_h": dict(decay_mult=0.0),
            "rel_pos_w": dict(decay_mult=0.0),
        },
    ),
)
# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type="LinearLR",
        start_factor=1e-6,
        by_epoch=True,
        begin=0,
        end=5,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(
        type="CosineAnnealingLR",
        T_max=95,
        by_epoch=True,
        begin=5,
        end=100,
    ),
]

# training schedule for 100 epochs
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=100, val_interval=interval)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=interval,
        save_best="mIoU",
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="CDVisualizationHook", draw=True, interval=1),
)

log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)


############################### running schedule #################################

# model settings
norm_cfg = dict(type="SyncBN", requires_grad=True)

data_preprocessor = dict(
    type="DualInputSegDataPreProcessor",
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32),
)
model = dict(
    type="SiamEncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="arm_base_pz16",
        # original 800
        img_size=size,
        drop_rate=0,
        drop_path_rate=0.1,
        pretrained="file:///home/chenmingshuo/projects/Mamba/MTP/Mamba_ckpts/final-base-400w-400epoch.pth",
    ),
    neck=dict(type="FeatureFusionNeck", policy="abs_diff", out_indices=(0, 1, 2, 3)),
    decode_head=dict(
        type="UNetHead",
        num_classes=2,
        ignore_index=255,
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        channels=64,
        dropout_ratio=0.1,
        encoder_channels=[768, 768, 768, 768],
        decoder_channels=[512, 256, 128, 64],
        n_blocks=4,
        use_batchnorm=True,
        center=False,
        attention_type=None,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type="mmseg.CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0
        ),
    ),
    # decode_head=dict(
    #     type='UPerHead',
    #     in_channels=[768, 768, 768, 768],
    #     num_classes=2,
    #     ignore_index=255,
    #     in_index=[0, 1, 2, 3],
    #     pool_scales=(1, 2, 3, 6),
    #     channels=768,
    #     dropout_ratio=0.1,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
    # ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)
# test_cfg=dict(mode='slide', stride=(112,112), crop_size=(224, 224)))
