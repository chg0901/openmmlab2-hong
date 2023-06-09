# _base_ = [
#     '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
#     '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
# ]


_base_ = ['./resnet50_fintune.py']
model = dict(

    head=dict(
        num_classes=30,
    ),
    init_cfg = dict(type='Pretrained',
                    checkpoint="https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth")
)



# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=30
)



train_dataloader = dict(
    batch_size=32,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        # ann_file=None,
        # data_prefix=None,
        data_root='../../data/fruit/train',),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        # ann_file=None,
        # data_prefix=None,
        data_root='../../data/fruit/val',),
)



val_evaluator = dict(type='Accuracy', topk=(1,5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(lr=0.01,))

# learning policy
param_scheduler = dict( milestones=[3, 6, 9])


# train, val, test setting
train_cfg = dict( max_epochs=10, val_interval=2)

# configure default hooks
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_best='auto'),
)

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=23, deterministic=False)



