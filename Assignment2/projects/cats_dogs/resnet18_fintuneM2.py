# _base_ = [
#     '../_base_/models/resnet18.py', '../_base_/datasets/imagenet_bs32.py',
#     '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
# ]


_base_ = ['./resnet18_fintune.py']
model = dict(

    head=dict(
        num_classes=2,
    ),
    init_cfg = dict(type='Pretrained',
                    checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet18_8xb32_in1k_20210831-fbbb1da6.pth')
)



# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=2
)



train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        # ann_file=None,
        # data_prefix=None,
        data_root='../../data/cats_dogs_dataset/training_set',),
)

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        # ann_file=None,
        # data_prefix=None,
        data_root='../../data/cats_dogs_dataset/training_set',),
)


optim_wrapper = dict(
    optimizer=dict(lr=0.01,))


train_cfg = dict(by_epoch=True, max_epochs=5, val_interval=2)


# configure default hooks
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=2, save_best='auto'),
)

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=23, deterministic=False)



