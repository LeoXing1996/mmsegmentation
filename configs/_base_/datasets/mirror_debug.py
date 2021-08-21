dataset_type = 'MirrorDataset'
data_root = 'data/ICIP_mirror'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
depth_norm_cfg = dict(mean=[808.18], std=[943.49])

img_scale = (640, 480)
crop_size = (480, 480)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # feel free to use this pipeline, if there is no depth, nothing happend.
    dict(type='LoadDepthMap'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # feel free to use this pipeline, if there is no depth, nothing happend.
    dict(type='NormalizeDepth', **depth_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    # If you want to use depth map, you should add `depth_map` in this term
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'depth_map']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthMap'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='NormalizeDepth', **depth_norm_cfg),
            # dict(type='ImageToTensor', keys=['img', 'depth_map']),
            dict(type='ImageToTensor', keys=['img', 'depth_map']),
            # If you want to use depth map,
            # you should add `depth_map` in this term
            dict(type='Collect', keys=['img', 'depth_map']),
        ])
]

# no split
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/rgb',
        ann_dir='train/mask',
        depth_dir='train/depth',
        split='train/train_name.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/rgb',
        ann_dir='train/mask',
        depth_dir='train/depth',
        split='train/debug.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        # img_dir='stage_one/rgb',
        img_dir='train/rgb',
        ann_dir='train/mask',
        split='train/val_name.txt',
        pipeline=test_pipeline,
        trans_from_ade=True))
