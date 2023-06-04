# 测试结果

## RTMDet-tiny
best epoch: 196/200
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.808
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.970
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.970
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.808
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.838
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.838
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.838
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.838
06/04 16:54:47 - mmengine - INFO - bbox_mAP_copypaste: 0.808 0.970 0.970 -1.000 -1.000 0.808
06/04 16:54:47 - mmengine - INFO - Epoch(test) [3/3]    coco/bbox_mAP: 0.8080  coco/bbox_mAP_50: 0.9700  coco/bbox_mAP_75: 0.9700  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.8080  data_time: 1.4042  time: 1.5282
```

RTMPose-s

best epoch: 255/300
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.741
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.969
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.741
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.781
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.976
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.781
06/04 20:49:49 - mmengine - INFO - Evaluating PCKAccuracy (normalized by ``"bbox_size"``)...
06/04 20:49:49 - mmengine - INFO - Evaluating AUC...
06/04 20:49:49 - mmengine - INFO - Evaluating NME...
06/04 20:49:49 - mmengine - INFO - Epoch(test) [6/6]    coco/AP: 0.740501  coco/AP .5: 1.000000  coco/AP .75: 0.968647  coco/AP (M): -1.000000  coco/AP (L): 0.740501  coco/AR: 0.780952  coco/AR .5: 1.000000  coco/AR .75: 0.976190  coco/AR (M): -1.000000  coco/AR (L): 0.780952  PCK: 0.975057  AUC: 0.137925  NME: 0.040603  data_time: 1.145144  time: 1.170622
```

# 测试可视化
## 测试图片和视频
单张图片和两个测试视频

![Ear.jpg](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/MyEar/Ear.jpg)

[Ear.mp4](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/MyEar/Ear.mp4)

[Ear2.mp4](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/MyEar/Ear2.mp4)


输出结果如下

## RTMDet-tiny
Ear Img rtmdet 可视化
![Ear.jpg](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/MyEar/E2_rtmdet/vis/Ear.jpg)

[Ear.mp4 rtmdet 结果可视化](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/MyEar/E2_rtmdet/Ear_pred0.6.mp4)

## RTMPose-s
Ear Img OpenCV 可视化
![Ear_pose OpenCV](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/MyEar/Ear_pose.jpg)

Ear Img Visualizer 可视化
![Ear_pose_visualizer](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/MyEar/Ear_pose_visualizer.jpg)

Ear Img RTMDet-RTMPose API 可视化
![Ear_pose2](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/MyEar/RTMDet-RTMPose/Ear.jpg)


[RTMPose-s Ear.mp4](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/MyEar/RTMDet-RTMPose/Ear.mp4)

[RTMPose-s Ear2.mp4](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/MyEar/RTMDet-RTMPose/Ear2.mp4)



# 代码Notebook
- [01Ear目标检测-训练RTMDet.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/01Ear%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B-%E8%AE%AD%E7%BB%832.ipynb)
- [02Ear目标检测-可视化训练日志.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/02Ear%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B-%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E6%97%A5%E5%BF%97.ipynb)
- [03Ear目标检测-模型权重文件精简转换.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/03Ear%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B-%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E6%96%87%E4%BB%B6%E7%B2%BE%E7%AE%80%E8%BD%AC%E6%8D%A2.ipynb)
- [04Ear目标检测-预测.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/04Ear%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B-%E9%A2%84%E6%B5%8B.ipynb)
- [05Ear关键点检测-训练RTMPose.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/05Ear%E5%85%B3%E9%94%AE%E7%82%B9%E6%A3%80%E6%B5%8B-%E8%AE%AD%E7%BB%83RTMPose.ipynb)
- [06Ear关键点检测-可视化训练日志.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/06Ear%E5%85%B3%E9%94%AE%E7%82%B9%E6%A3%80%E6%B5%8B-%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E6%97%A5%E5%BF%97.ipynb)
- [07Ear关键点检测预测-命令行.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/07Ear%E5%85%B3%E9%94%AE%E7%82%B9%E6%A3%80%E6%B5%8B%E9%A2%84%E6%B5%8B-%E5%91%BD%E4%BB%A4%E8%A1%8C.ipynb)
- [08Ear关键点检测预测-Python API.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/No1-Assignment/08Ear%E5%85%B3%E9%94%AE%E7%82%B9%E6%A3%80%E6%B5%8B%E9%A2%84%E6%B5%8B-Python%20API.ipynb)




# 训练权重

RTMDet-tiny模型权重
[best_coco_bbox_mAP_epoch_196.pth](https://drive.google.com/file/d/1oBaGq98r5VySlMtCbLCavkStpmgxNLYl/view?usp=drive_link)

RTMPose-s模型权重
[best_PCK_epoch_255.pth](https://drive.google.com/file/d/1lFYuInSq_YISKDQp1PAhuaLUMRMxLsos/view?usp=sharing)

对应的模型轻量化转换权重

[best_coco_bbox_mAP_epoch_196_zip-dc2ee3bc.pth](https://drive.google.com/file/d/12NlnTy3P7-oVaMaYev3xr3tZbb2zFC6l/view?usp=sharing)

[best_PCK_epoch_255_zip-d1bf22ba.pth](https://drive.google.com/file/d/1WQRyWI22EJji4IwI9M_1ye_ljUwuAIyq/view?usp=sharing)

# 笔记链接
1. [【七班】MMPose代码实践与耳朵穴位数据集实战【OpenMMLab AI实战营第二期Day3】](https://zhuanlan.zhihu.com/p/634511756) 
2. [【七班】人体姿态估计概论笔记【OpenMMLab AI实战营第二期Day2】](https://blog.csdn.net/chenghong1/article/details/131006094)
3. [【七班】OpenMMLab 开源项目介绍【OpenMMLab AI实战营第二期Day1】](https://blog.csdn.net/chenghong1/article/details/130988224)



------------------------------------------------------------------------------------------
以下为TODO备注内容
------------------------------------------------------------------------------------------


【A0】关于本教程.ipynb
【A1】安装MMPose.ipynb
【A2】安装MMDetection.ipynb
【B1】MMPose预训练模型预测-命令行.ipynb
【B2】MMPose预训练模型预测-Python API.ipynb
【C】下载三角板关键点检测数据集.ipynb
【D1】三角板目标检测-下载config配置文件.ipynb
【D2】三角板目标检测-训练.ipynb
【D3】三角板目标检测-可视化训练日志.ipynb
【D4】三角板目标检测-模型权重文件精简转换.ipynb
【E1】三角板目标检测-下载训练好的模型权重.ipynb
【E2】三角板目标检测-预测.ipynb
