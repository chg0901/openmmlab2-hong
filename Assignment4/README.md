# Assignment 4 and Tutorial
## Watermelon pixel-level semantic segmentation based on PSPNet

 [[Description]](https://github.com/open-mmlab/OpenMMLabCamp/issues/388)
[[Data 1 (raw data)]](https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/watermelon/Watermelon87_Semantic_Seg_Labelme.zip)
[[Data 2 (precessed data)]](https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/watermelon/Watermelon87_Semantic_Seg_Mask.zip)
[[Code Base]](https://github.com/TommyZihao/MMSegmentation_Tutorials/tree/main/20230612/%E3%80%90C1%E3%80%91Kaggle%E5%AE%9E%E6%88%98-%E8%BF%AA%E6%8B%9C%E5%8D%AB%E6%98%9F%E8%88%AA%E6%8B%8D%E5%A4%9A%E7%B1%BB%E5%88%AB%E8%AF%AD%E4%B9%89%E5%88%86%E5%89%B2)
[[Video1]](https://www.bilibili.com/video/BV1gV4y1m74P/) [[Video2]](https://www.bilibili.com/video/BV1uh411T73q/)

## Environment Setup

```
cd code_path

git clone https://github.com/open-mmlab/mmsegmentation.git -b dev-1.x

cd mmsegmentation
pip install -e .
```

## Organization of important files 

```
.
├── code_path
│   ├── mmsegmentation
│   │   ├── configs
│   │   │   ├── _base_
│   │   │   │   ├── datasets
│   │   │   │   │   ├── DubaiDataset_pipeline.py
│   │   │   │   │   └── Watermelon87Dataset_pipeline.py
│   │   │   │   ├── default_runtime.py
│   │   │   │   ├── models
│   │   │   │   │   └── pspnet_r50-d8.py
│   │   │   │   └── schedules
│   │   │   │       └── schedule_40k.py
│   │   │   └── pspnet
│   │   │       ├── pspnet_r50-d8_4xb2-40k_DubaiDataset.py
│   │   │       └── pspnet_r50-d8_4xb2-40k_Watermelon87Dataset.py
│   │   ├── data
│   │   │   ├── Watermelon87_Semantic_Seg_Mask
│   │   ├── mmseg
│   │   │   └── datasets
│   │   │       ├── DubaiDataset.py
│   │   │       ├── __init__.py
│   │   │       ├── StanfordBackgroundDataset.py
│   │   │       └── Watermelon87Dataset.py
... ...
│   └── MMSeg_tutorial
│   │   ├── 【A】安装配置MMSegmentation.ipynb
│   │   ├── 【B1】预训练语义分割模型预测-单张图像-命令行.ipynb
│   │   ├── 【B3】预训练语义分割模型预测-视频.ipynb
│   │   ├── 【C1】Kaggle实战-迪拜卫星航拍多类别语义分割
│   │   ├── 【C2】Kaggle实战-小鼠肾小球组织病理切片语义分割
│   │   ├── 【Z1】扩展阅读.ipynb
│   │   ├── 【Z2】MMSegmentation代码实战作业.ipynb
│   │   └── 【作业】西瓜多类别语义分割
│   │       ├── 【A】下载整理好的数据集.ipynb
│   │       ├── 【B】可视化探索数据集.ipynb
│   │       ├── 【C】准备config配置文件2.ipynb
│   │       ├── 【D】MMSeg训练语义分割模型-Copy1.ipynb
│   │       ├── 【E】可视化训练日志.ipynb
│   │       ├── 【F】用训练得到的模型预测-Copy1.ipynb
│   │       ├── 【G】测试集性能评估-Copy1.ipynb
│   │       ├── 【H1】语义分割模型预测-单张图像-命令行.ipynb
│   │       ├── 【H2】语义分割模型预测-视频.ipynb
│   │       └── 【H3】Test_with_python_file_on_Documents.ipynb
│   ├── good_codes [From https://github.com/zeyuanyin/OpenMMLabCamp/tree/main/homework-4]
│   │   ├── log
│   │   ├── modify_config.py
│   │   ├── pspnet-Watermelon.py
│   │   ├── run.py
│   │   ├── test_img
│   │   ├── test_img.py
│   │   └── test_video.py

```

## Fix data name error

rename:
```
./mmsegmentation/data/Watermelon87_Semantic_Seg_Mask/img_dir/train/21746.1.jpg ->  21746.jpg
./mmsegmentation/data/Watermelon87_Semantic_Seg_Mask/img_dir/val/01bd15599c606aa801201794e1fa30.jpg@1280w_1l_2o_100sh.jpg -> 01bd15599c606aa801201794e1fa30.jpg

```


## Dataset Registeration and Config Modification  
Notebook:[**【作业】西瓜多类别语义分割/【C】准备config配置文件2.ipynb**](./MMSeg_tutorial/【作业】西瓜多类别语义分割/【C】准备config配置文件2.ipynb)

#### 1. Define the Dataset Class (class name and color)
- Download `DubaiDataset.py` and Copy it as `Watermelon87Dataset.py`
```
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/Dubai/DubaiDataset.py -P mmseg/datasets
```
- Modify the `classes` in `./mmsegmentation/mmseg/datasets/Watermelon87Dataset.py` to `'classes':['Red', 'Green', 'White', 'Seed-black', 'Seed-white', 'Unlabeled']`
```python
# %load mmseg/datasets/Watermelon87Dataset.py
# 同济子豪兄 2023-2-15
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class Watermelon87Dataset(BaseSegDataset):
    # 类别和对应的可视化配色
    METAINFO = {
        'classes':['Red', 'Green', 'White', 'Seed-black', 'Seed-white', 'Unlabeled'],
        'palette':[[132,41,246], [228,193,110], [152,16,60], [58,221,254], [41,169,226], [155,155,155]]
    }
    
    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False, # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
```

#### 2. Register the dataset class
- Download `__init__.py` and add related class related to `Watermelon87Dataset` and  `Watermelon87Dataset.py`
```
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/Dubai/__init__.py -P mmseg/datasets
```
**key modification**
```python
from .Watermelon87Dataset import Watermelon87Dataset
__all__ = [ 'Watermelon87Dataset']

```

#### 3. Define the Dataset pipline for training and test
- Download `DubaiDataset_pipeline.py` and Copy it as `Watermelon87Dataset_pipeline.py`
```
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/Dubai/DubaiDataset_pipeline.py -P configs/_base_/datasets
```
- Change the `dataset_type` and `data_root` 
Modify the config file to fit the Watermelon dataset, add your data address
```python
# %load configs/_base_/datasets/Watermelon87Dataset_pipeline.py
# dataset settings
dataset_type = 'Watermelon87Dataset' # 数据集类名
data_root = 'data/Watermelon87_Semantic_Seg_Mask' # 数据集路径（相对于mmsegmentation主目录）
# ......
```

#### 4. Download  and modify the Config document
- Download `pspnet_r50-d8_4xb2-40k_DubaiDataset.py.py`  and copy it as `./configs/pspnet/pspnet_r50-d8_4xb2-40k_Watermelon87Dataset.py`
```
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/Dubai/pspnet_r50-d8_4xb2-40k_DubaiDataset.py -P configs/pspnet 
```

Look at this config file, we need to copy the `pspnet_r50-d8.py`, `default_runtime.py`, `schedule_40k.py` from `mmsegmentation/configs/_base_/` to the current folder with the same structure path.

```python
_base_ = [
    '../_base_/models/pspnet_r50-d8.py', '../_base_/datasets/Watermelon87Dataset_pipeline.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (64, 64) # 输入图像尺寸，根据自己数据集情况修改
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
```
**load the config file**
```python
from mmengine import Config
cfg = Config.fromfile('./configs/pspnet/pspnet_r50-d8_4xb2-40k_Watermelon87Dataset.py')
```
**key modification**
```python
# 结果保存目录
cfg.work_dir = './work_dirs/Watermelon87_Semantic_Seg_Mask'

cfg['randomness'] = dict(seed=42)
cfg.optimizer.lr=0.005
cfg.optim_wrapper.optimizer = cfg.optimizer
cfg.train_pipeline[3].crop_size = cfg.crop_size
cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline

cfg.default_hooks.checkpoint.max_keep_ckpts=2
cfg.default_hooks.checkpoint.save_best=['mAcc','mIoU']  # auto
cfg.default_hooks.checkpoint.rule= 'greater'
```
**process the modification for config**
```
# python ./good_codes/modify_config.py
```
The modified config file is saved at `./mmsegmentation/pspnet-Watermelon87_Semantic_Seg_Mask_20230616.py`.

## Training and training logs
[./MMSeg_tutorial/【作业】西瓜多类别语义分割/【D】MMSeg训练语义分割模型-Copy1.ipynb](./MMSeg_tutorial/【作业】西瓜多类别语义分割/【D】MMSeg训练语义分割模型-Copy1.ipynb)
```python
# python run.py  # ./good_codes/run.py
```

Outputs:

[./work_dirs/Watermelon87_Semantic_Seg_Mask/20230616_114954/20230616_114954.log](./work_dirs/Watermelon87_Semantic_Seg_Mask/20230616_114954/20230616_114954.log)
```
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
|    Red     | 92.13 | 96.03 |
|   Green    | 87.87 | 91.65 |
|   White    | 77.85 | 91.88 |
| Seed-black | 84.83 | 93.94 |
| Seed-white | 68.25 | 73.52 |
| Unlabeled  | 52.67 |  73.6 |
+------------+-------+-------+
2023/06/16 12:09:59 - mmengine - INFO - Iter(val) [11/11]  
aAcc: 94.2700  mIoU: 77.2600  mAcc: 86.7700  data_time: 0.0016  time: 0.2658

2023/06/16 12:09:59 - mmengine - INFO - The previous best checkpoint /home/cine/Documents/GitHub/mmsegmentation/work_dirs/Watermelon87_Semantic_Seg_Mask/best_mAcc_iter_3600.pth is removed
2023/06/16 12:10:00 - mmengine - INFO - The best checkpoint with 86.7700 mAcc at 7600 iter is saved to best_mAcc_iter_7600.pth.
2023/06/16 12:10:00 - mmengine - INFO - The previous best checkpoint /home/cine/Documents/GitHub/mmsegmentation/work_dirs/Watermelon87_Semantic_Seg_Mask/best_mIoU_iter_2800.pth is removed
2023/06/16 12:10:01 - mmengine - INFO - The best checkpoint with 77.2600 mIoU at 7600 iter is saved to best_mIoU_iter_7600.pth.

```

## Test and test log

Notebook:[./MMSeg_tutorial/【作业】西瓜多类别语义分割/【H3】Test_with_python_file_on_Documents.ipynb](./MMSeg_tutorial/【作业】西瓜多类别语义分割/【H3】Test_with_python_file_on_Documents.ipynb)

```
python mmsegmentation/tools/test.py \
       pspnet-Watermelon.py \
       ../log/Watermelon/iter_3000.pth \
       --work-dir ../log/test

```
Outputs is the same as the above train outputs since we test evey 400 epoch during training.
[./mmsegmentation/work_dirs/Watermelon87_Semantic_Seg_Mask/20230616_160732/20230616_160732.log](./mmsegmentation/work_dirs/Watermelon87_Semantic_Seg_Mask/20230616_160732/20230616_160732.log)

## Inference on Watermelon Image
Notebook:[【H1】语义分割模型预测-单张图像-命令行.ipynb](./MMSeg_tutorial/【作业】西瓜多类别语义分割/【H1】语义分割模型预测-单张图像-命令行.ipynb)

Notebook:[H3】Test_with_python_file_on_Documents.ipynb](./MMSeg_tutorial/【作业】西瓜多类别语义分割/【H3】Test_with_python_file_on_Documents.ipynb)
``` python
# python test_img.py
# ./good_codes/test_img.py

checkpoint_path = 'work_dirs/Watermelon87_Semantic_Seg_Mask/best_mAcc_iter_7600.pth'
# img_path = './data/watermelon.png'
img_path = './data/watermelon_2.png'

# Command 
python demo/image_demo.py \
        data/watermelon.png \
        pspnet-Watermelon87_Semantic_Seg_Mask_20230616.py\
        work_dirs/Watermelon87_Semantic_Seg_Mask/best_mAcc_iter_7600.pth \
        --out-file outputs/watermelon_pred.png \
        --device cuda:0 \
        --opacity 0.5
        
python demo/image_demo.py \
        data/watermelon_2.png \
        pspnet-Watermelon87_Semantic_Seg_Mask_20230616.py\
        work_dirs/Watermelon87_Semantic_Seg_Mask/best_mAcc_iter_7600.pth \
        --out-file outputs/watermelon_2_pred.png \
        --device cuda:0 \
        --opacity 0.5
        
```


<div align=left>
<img width=30% src="./mmsegmentation/data/watermelon_2.png"/>
<img width=30% src="./mmsegmentation/data/watermelon_2_pred.png"/>
<br>
<img width=30% src="./mmsegmentation/data/watermelon.png"/>
<img width=30% src="./mmsegmentation/data/watermelon_pred.png"/>
</div>



## Inference on Watermelon Video

Video Source: https://www.youtube.com/watch?v=yFPB5wct9KU

Notebook:[【H3】Test_with_python_file_on_Documents.ipynb](./MMSeg_tutorial/【作业】西瓜多类别语义分割/【H3】Test_with_python_file_on_Documents.ipynb)
```
# Fast
# python test_video.py
# ./good_code/test_video.py
Assignment4/MMSeg_tutorial/【作业】西瓜多类别语义分割/【H2】语义分割模型预测-视频.ipynb
```

<div align=left>
<img width=30% src="./mmsegmentation/data/watermelon2.gif"/>
<img width=30% src="./mmsegmentation/data/watermelon_pred2_.gif"/>
</div>


**Gif 生成方法(基于linux ffmpeg)**
```python
sudo snap install ffmpeg
sudo apt install yasm

ffmpeg -i ./data/watermelon.mp4 ./data/watermelon2.gif
ffmpeg -i ./data/watermelon_pred_2.mp4 ./data/watermelon_pred2_.gif
```


```
# Very slow with demo/image_demo.py （NOT Recommend）

!python demo/image_demo.py \
        data/watermelon_2.png \
        pspnet-Watermelon87_Semantic_Seg_Mask_20230616.py\
        work_dirs/Watermelon87_Semantic_Seg_Mask/best_mAcc_iter_7600.pth \
        --out-file outputs/watermelon_2_pred.png \
        --device cuda:0 \
        --opacity 0.3
```

# Reference

[https://github.com/zeyuanyin/OpenMMLabCamp/tree/main/homework-4](https://github.com/zeyuanyin/OpenMMLabCamp/tree/main/homework-4)

this repo helps me a lot in better writing a clear readme file and file organization of assignment files