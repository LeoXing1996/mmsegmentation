_base_ = [
    '../_base_/models/mirrornet.py', '../_base_/datasets/mirror_debug.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    # use `with_depth` for depth input
    decode_head=dict(with_depth=True),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
work_dir = 'work_dirs/debug'
evaluation = dict(
    interval=1,
    metric='mIoU',
    do_vis=True,
    vis_cfg=dict(vis_depth=True, max_num=2))
