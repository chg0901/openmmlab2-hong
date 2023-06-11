**作业**：基于 RTMDet 的气球检测

**背景**：熟悉目标检测和 MMDetection 常用自定义流程。

**任务**：

1. **更换balloon气球数据集**：基于提供的 notebook，将 cat 数据集换成气球数据集
2. 按照视频中 notebook 步骤，**可视化数据集和标签**
3. 使用**MMDetection**算法库，**训练 RTMDet 气球目标检测模型**，可以**适当调参**，**提交测试集评估指标**
4. 用网上下载的任意包括气球的图片进行**预测**，
5. 按照视频中 notebook 步骤，对 demo 图片进行**特征图可视化**和 **Box AM 可视化**
 


- 将**预测结果保存到github仓库中，readme中呈现出来**
- **需提交的测试集评估指标（不能低于baseline指标的50%）**

        目标检测 RTMDet-tiny 模型性能 准确率65 mAP

**数据集**：[**气球数据集**](https://download.openmmlab.com/mmyolo/data/balloon_dataset.zip) 

> 同时也欢迎各位选择更复杂的数据集进行训练，可能会有额外加分哟
> 如选用同济子豪兄的 [十类饮料目标检测数据集Drink_284 ](https://github.com/TommyZihao/Train_Custom_Dataset/tree/main/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E6%95%B0%E6%8D%AE%E9%9B%86)

# 任务完成
## 1. 更换数据集
采用了【同济子豪兄的 [十类饮料目标检测数据集Drink_284](https://link.zhihu.com/?target=https%3A//github.com/TommyZihao/Train_Custom_Dataset/tree/main/%25E7%259B%25AE%25E6%25A0%2587%25E6%25A3%2580%25E6%25B5%258B/%25E7%259B%25AE%25E6%25A0%2587%25E6%25A3%2580%25E6%25B5%258B%25E6%2595%25B0%25E6%258D%25AE%25E9%259B%2586)】中的[MS COCO标注格式（已划分训练集和测试集）](https://link.zhihu.com/?target=https%3A//zihao-download.obs.cn-east-3.myhuaweicloud.com/yolov8/datasets/Drink_284_Detection_Dataset/Drink_284_Detection_coco.zip)

### **Config 中对数据集的配置** 

**训练配置**[rtmdet_tiny_1xb12-40e_drinks.py](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment3/mmdetection/rtmdet_tiny_1xb12-40e_drinks.py) 

```python 
data_root = './data/Drink_284_Detection_coco/'

# 非常重要
metainfo = {
    'classes': ("cola","pepsi","sprite","fanta","spring", 
                "ice", "scream","milk","red","king"),
    'palette': [
        (101, 205, 228),(240, 128, 128), (154, 205, 50), (34, 139, 34), 
        (139, 0, 0), (255, 165, 0), (255, 0, 255), (255, 255, 0), 
        (29, 123, 243), (0, 255, 255), 
    ]
}
num_classes = 10

```


## 2. 可视化数据集和标签
- **数据集可视化**
![image](https://github.com/chg0901/openmmlab2-hong/assets/8240984/8413ed9d-5274-43b4-85ee-51f40dfb4084)
- **COCO Json 可视化**
![image](https://github.com/chg0901/openmmlab2-hong/assets/8240984/d7b4300a-2e19-4102-9fef-3abb8cdef246)
- **训练前可视化验证**
![image](https://github.com/chg0901/openmmlab2-hong/assets/8240984/29cf2838-662a-4d02-82e5-9add6de8cb38)

## 3. 训练与测试评估

- [train log](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment3/mmdetection/work_dirs/rtmdet_tiny_1xb12-40e_drinks/20230610_141823/20230610_141823.log)
- [test log](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment3/mmdetection/work_dirs/rtmdet_tiny_1xb12-40e_drinks/20230610_144158/20230610_144158.log)

- coco/bbox_mAP: 0.9550
- coco/bbox_mAP_50: 0.9920
- coco/bbox_mAP_75: 0.9920
- coco/bbox_mAP_l: 0.9550

```
06/10 14:42:07 - mmengine - INFO - Load checkpoint from work_dirs/rtmdet_tiny_1xb12-40e_drinks/best_coco/bbox_mAP_epoch_35.pth
....
DONE (t=0.08s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.955
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.992
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.992
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.955
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.940
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.969
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.969
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.969
06/10 14:44:28 - mmengine - INFO - bbox_mAP_copypaste: 0.955 0.992 0.992 -1.000 -1.000 0.955
06/10 14:44:28 - mmengine - INFO - Epoch(test) [56/56]  coco/bbox_mAP: 0.9550  coco/bbox_mAP_50: 0.9920  coco/bbox_mAP_75: 0.9920  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.9550  data_time: 2.4391  time: 2.4667
```

## 4. 网上照片预测与评估
- [pepsi_test2可乐可视化](https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment3/mmyolo/output/pepsi_test)
- [原图文件夹](https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment3/mmdetection/data)
- [预测文件夹](https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment3/mmyolo/output)

####  原图
![image](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment3/mmdetection/data/pepsi_test2.jpeg)
####  预测图
![image](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment3/mmyolo/output/pepsi_test2.jpeg)
### 可视化
####  pepsi_test2_backbone.jpeg
![image](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment3/mmyolo/output/pepsi_test/pepsi_test2_backbone.jpeg)
####  pepsi_test2_neck.jpeg
![image](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment3/mmyolo/output/pepsi_test/pepsi_test2_neck.jpeg)
####  pepsi_test2_neck.out_convs[2]
![image](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment3/mmyolo/output/pepsi_test/pepsi_test2_neck.out_convs%5B0%5D.jpeg)
####  pepsi_test2_neck.out_convs[0]
![image](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment3/mmyolo/output/pepsi_test/pepsi_test2_neck.out_convs%5B2%5D.jpeg)
