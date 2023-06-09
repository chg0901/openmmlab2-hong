# Fruits数据集作业目录
https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment2/projects/fruits

**题目：** 基于 ResNet50 的水果分类

**背景：** 使用基于卷积的深度神经网络 ResNet50 对 30 种水果进行分类

**任务**

1. 划分训练集和验证集
2. 按照 MMPreTrain CustomDataset 格式组织训练集和验证集
3. 使用 MMPreTrain 算法库，编写配置文件，正确加载预训练模型
4. 在水果数据集上进行微调训练
5. 使用 MMPreTrain 的 ImageClassificationInferencer 接口，对网络水果图像，或自己拍摄的水果图像，使用训练好的模型进行分类
**需提交的验证集评估指标（不能低于 60%）**
**ResNet-50**
![image](https://github.com/chg0901/openmmlab2-hong/assets/8240984/52de578e-42d1-45ab-87c6-ea0c60e59044)

# 任务完成
## 1. [**dataset_process.py**](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment2/projects/fruits/dataset_process.py)
## 2/3/4. **Config 中对数据集的配置** 

 
**训练配置[projects/fruits/resnet50_fintuneM2.py](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment2/projects/fruits/resnet50_fintuneM2.py) 继承自[projects/fruits/resnet50_fintune.py](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment2/projects/fruits/resnet50_fintune.py)**

```python 
# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(num_classes=30)

train_dataloader = dict(
    batch_size=32,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        data_root='../../data/fruit/train',),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=12,
    dataset=dict(
        type=dataset_type,
        data_root='../../data/fruit/val',),
)

val_evaluator = dict(type='Accuracy', topk=(1,5))

test_dataloader = val_dataloader
test_evaluator = val_evaluator

```

### 完整数据集Config


``` python
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=30,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs')
]
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=12,
    dataset=dict(
        type='CustomDataset',
        data_root='../../data/fruit/train',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='RandomResizedCrop', scale=224),
            dict(type='RandomFlip', prob=0.5, direction='horizontal'),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    num_workers=12,
    dataset=dict(
        type='CustomDataset',
        data_root='../../data/fruit/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = dict(type='Accuracy', topk=(1, 5))
test_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=64,
    num_workers=12,
    dataset=dict(
        type='CustomDataset',
        data_root='../../data/fruit/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_evaluator = dict(type='Accuracy', topk=(1, 5))

```

[**完整Config配置(projects/fruits/exp3_resnet50/resnet50_fintuneM2.py) **](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment2/projects/fruits/exp3_resnet50/resnet50_fintuneM2.py)

### 测试结果

- **accuracy/top1: 91.5138**
- **accuracy/top5: 98.3945**

[**测试日志log**20230608_171139.log](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment2/projects/fruits/work_dirs/resnet50_fintuneM2/20230608_171139/20230608_171139.log)

```
2023/06/08 17:11:45 - mmengine - WARNING - The prefix is not set in metric class DumpResults.
2023/06/08 17:11:46 - mmengine - INFO - Load checkpoint from exp3_resnet50/best_accuracy_top1_epoch_8.pth
2023/06/08 17:11:47 - mmengine - INFO - Results has been saved to result_resnet50.pkl.
2023/06/08 17:11:47 - mmengine - INFO - Epoch(test) [7/7]    accuracy/top1: 91.5138  accuracy/top5: 98.3945  data_time: 0.0964  time: 0.2016
```
### 自己数据集的测试
**测试图片文件夹**   [Assignment2/data](https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment2/data)
![all_test_image.png](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment2/projects/fruits/all_test_image.png)

**测试脚本文件**     [projects/fruits/**resnet50_fintune_test.py**](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment2/projects/fruits/resnet50_fintune_test.py)

**生成图片文件夹**   [projects/fruits/outputs_test_data](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment2/projects/fruits/outputs_test_data/)
![all_predicted_image.png](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment2/projects/fruits/all_predicted_image.png)


```python
from mmpretrain import ImageClassificationInferencer
import cv2
import matplotlib.pyplot as plt
inferencer = ImageClassificationInferencer('./resnet50_fintuneM2.py',
                                           pretrained='./exp3_resnet50/best_accuracy_top1_epoch_8.pth')

image_list = ['data/apple.jpeg', 'data/banana.jpeg', 'data/fruit.jpeg', 'data/grapes.jpg']
image_list = ['../../'+i for i in image_list]

fig, _ = plt.subplots(nrows=2, ncols=2,figsize=(15, 15))
for i, ax in enumerate(fig.axes):
    img = cv2.imread(image_list[i])[:, :, (2, 1, 0)]
    # img=cv2.imread('outputs_test_data/apple.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.axis("off")
    ax.imshow(img)

plt.savefig("all_test_image.png", bbox_inches='tight')
plt.show()

# infer all image in the image_list
results = inferencer(image_list, batch_size=4)

print_keys = list(results[0].keys())

fig, _ = plt.subplots(nrows=2, ncols=2,figsize=(15, 15))
for i, ax in enumerate(fig.axes):
    image_name = image_list[i].split('/')[-1].split('.')[0]
    img = cv2.imread(f'outputs_test_data/{image_name}.png')[:, :, (2, 1, 0)]
    # img=cv2.imread('outputs_test_data/apple.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    ax.axis("off")
    ax.imshow(img)

    print(f"file name: {image_list[i]}")
    print(f"{print_keys[1]}: {results[i][print_keys[1]]}")
    print(f"{print_keys[3]}: {results[i][print_keys[3]]}")
    print()

plt.savefig("all_predicted_image.png", bbox_inches='tight')
plt.show()

```

#### 训练日志

[训练日志log:   20230608_170620.log](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment2/projects/fruits/exp3_resnet50/20230608_170620/20230608_170620.log)
```
2023/06/08 17:06:36 - mmengine - INFO - Checkpoints will be saved to /home/cine/Documents/GitHub/mmpretrain/projects/fruits/exp3_resnet50.
2023/06/08 17:06:54 - mmengine - INFO - Epoch(train)  [1][100/109]  lr: 1.0000e-02  eta: 0:03:01  time: 0.1751  data_time: 0.0004  memory: 2965  loss: 1.2929
2023/06/08 17:06:56 - mmengine - INFO - Exp name: resnet50_fintuneM2_20230608_170620
2023/06/08 17:06:56 - mmengine - INFO - Saving checkpoint at 1 epochs
2023/06/08 17:07:15 - mmengine - INFO - Epoch(train)  [2][100/109]  lr: 1.0000e-02  eta: 0:02:38  time: 0.1759  data_time: 0.0004  memory: 2965  loss: 1.0736
2023/06/08 17:07:16 - mmengine - INFO - Exp name: resnet50_fintuneM2_20230608_170620
2023/06/08 17:07:16 - mmengine - INFO - Saving checkpoint at 2 epochs
2023/06/08 17:07:19 - mmengine - INFO - Epoch(val) [2][7/7]    accuracy/top1: 71.3303  accuracy/top5: 93.3486  data_time: 0.0976  time: 0.1949
2023/06/08 17:07:19 - mmengine - INFO - The best checkpoint with 71.3303 accuracy/top1 at 2 epoch is saved to best_accuracy_top1_epoch_2.pth.
2023/06/08 17:07:39 - mmengine - INFO - Epoch(train)  [3][100/109]  lr: 1.0000e-02  eta: 0:02:18  time: 0.1766  data_time: 0.0004  memory: 2965  loss: 0.7937


.............

2023/06/08 17:09:29 - mmengine - INFO - Exp name: resnet50_fintuneM2_20230608_170620
2023/06/08 17:09:29 - mmengine - INFO - Saving checkpoint at 8 epochs
2023/06/08 17:09:31 - mmengine - INFO - Epoch(val) [8][7/7]    accuracy/top1: 91.5138  accuracy/top5: 98.3945  data_time: 0.0275  time: 0.1280
2023/06/08 17:09:31 - mmengine - INFO - The previous best checkpoint /home/cine/Documents/GitHub/mmpretrain/projects/fruits/exp3_resnet50/best_accuracy_top1_epoch_6.pth is removed
2023/06/08 17:09:32 - mmengine - INFO - The best checkpoint with 91.5138 accuracy/top1 at 8 epoch is saved to best_accuracy_top1_epoch_8.pth.
2023/06/08 17:09:51 - mmengine - INFO - Epoch(train)  [9][100/109]  lr: 1.0000e-04  eta: 0:00:21  time: 0.1805  data_time: 0.0004  memory: 2965  loss: 0.3272
2023/06/08 17:09:53 - mmengine - INFO - Exp name: resnet50_fintuneM2_20230608_170620
2023/06/08 17:09:53 - mmengine - INFO - Saving checkpoint at 9 epochs
2023/06/08 17:09:57 - mmengine - INFO - Exp name: resnet50_fintuneM2_20230608_170620
2023/06/08 17:10:12 - mmengine - INFO - Epoch(train) [10][100/109]  lr: 1.0000e-05  eta: 0:00:01  time: 0.1826  data_time: 0.0004  memory: 2965  loss: 0.2552
2023/06/08 17:10:13 - mmengine - INFO - Exp name: resnet50_fintuneM2_20230608_170620
2023/06/08 17:10:13 - mmengine - INFO - Saving checkpoint at 10 epochs
2023/06/08 17:10:15 - mmengine - INFO - Epoch(val) [10][7/7]    accuracy/top1: 90.8257  accuracy/top5: 98.8532  data_time: 0.0329  time: 0.1310
```


## 命令行工具
 **工作目录**：projects/cats_dogs$
```
# 训练  
 mim train mmpretrain resnet18_fintuneM.py --work-dir=./exp


# 以指定训练权重测试并结果日志保存为pkl备用
 mim test mmpretrain resnet18_fintuneM.py  --checkpoint exp/epoch_5.pth --out result.pkl


# 对测试集进行分析，生成success和fail文件夹分别存储成功的和不成功的预测结果
 mim run mmpretrain analyze_results resnet18_fintuneM.py  result.pkl  --out-dir analyze


# 生成混淆矩阵，可视化分析预测结果
 mim run mmpretrain confusion_matrix resnet18_fintuneM.py result.pkl  --show --include-values
```



# 笔记链接
1. [【七班】MMPose代码实践与耳朵穴位数据集实战【OpenMMLab AI实战营第二期Day3】](https://zhuanlan.zhihu.com/p/634511756) 
2. [【七班】人体姿态估计概论笔记【OpenMMLab AI实战营第二期Day2】](https://blog.csdn.net/chenghong1/article/details/131006094)
3. [【七班】OpenMMLab 开源项目介绍【OpenMMLab AI实战营第二期Day1】](https://blog.csdn.net/chenghong1/article/details/130988224)

