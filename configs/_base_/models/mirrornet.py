# model settings
model = dict(
    type='EncoderDecoder',
    # pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNeXt_Mirror',
        backbone_path=None),
    decode_head=dict(
        type='MirrorNet',
        in_channels=2048,
        channels=32,
        num_classes=3,
        tar_shape=384,
        loss_decode=dict(
            type='LovaszLoss', reduction='none', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

find_unused_parameters = True
