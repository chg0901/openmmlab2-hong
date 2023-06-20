# 测试结果

## RTMDet-tiny
best epoch: 196/200 
1. [Json sclar](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/mmdetection/work_dirs/rtmdet_tiny_Ear/20230604_153610_ear_train_200_best196_.808/vis_data/20230604_153610.json)
2. [Config file](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/mmdetection/work_dirs/rtmdet_tiny_Ear/20230604_153610_ear_train_200_best196_.808/vis_data/config.py)
3. [Training and Validation Log](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/mmdetection/work_dirs/rtmdet_tiny_Ear/20230604_153610_ear_train_200_best196_.808/20230604_153610.log)
```
coco/bbox_mAP: 0.8080     coco/bbox_mAP_50: 0.9700  coco/bbox_mAP_75: 0.9700  
coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.8080
```

## RTMPose-s

best epoch: 255/300
1. [Json sclar](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/mmpose/work_dirs/rtmpose-s-Ear/20230604_172709/vis_data/20230604_172709.json)
2. [Config file](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/mmpose/work_dirs/rtmpose-s-Ear/20230604_172709/vis_data/config.py)
3. [Training and Validation Log](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/mmpose/work_dirs/rtmpose-s-Ear/20230604_172709/20230604_172709.log)

```
coco/AP: 0.740501       coco/AP .5: 1.000000   coco/AP .75: 0.968647  coco/AP (M): -1.000000  coco/AP (L): 0.740501  
coco/AR: 0.780952       coco/AR .5: 1.000000   coco/AR .75: 0.976190  coco/AR (M): -1.000000  coco/AR (L): 0.780952 
PCK: 0.975057           AUC: 0.137925          NME: 0.040603 
```

# 测试可视化
## 测试图片和视频
单张图片和两个测试视频

![Ear.jpg](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/MyEar/Ear.jpg)

[Ear.mp4](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/MyEar/Ear.mp4)

[Ear2.mp4](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/MyEar/Ear2.mp4)


输出结果如下

## RTMDet-tiny
Ear Img rtmdet 可视化
![Ear.jpg](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/MyEar/E2_rtmdet/vis/Ear.jpg)

[Ear.mp4 rtmdet 结果可视化](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/MyEar/E2_rtmdet/Ear_pred0.6.mp4)

## RTMPose-s
Ear Img OpenCV 可视化
![Ear_pose OpenCV](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/MyEar/Ear_pose.jpg)

Ear Img Visualizer 可视化
![Ear_pose_visualizer](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/MyEar/Ear_pose_visualizer.jpg)

Ear Img RTMDet-RTMPose API 可视化
![Ear_pose2](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/MyEar/RTMDet-RTMPose/Ear.jpg)


[RTMPose-s Ear.mp4](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/MyEar/RTMDet-RTMPose/Ear.mp4)

[RTMPose-s Ear2.mp4](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/MyEar/RTMDet-RTMPose/Ear2.mp4)



# 代码Notebook
- [01Ear目标检测-训练RTMDet.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/01Ear%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B-%E8%AE%AD%E7%BB%832.ipynb)
- [02Ear目标检测-可视化训练日志.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/02Ear%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B-%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E6%97%A5%E5%BF%97.ipynb)
- [03Ear目标检测-模型权重文件精简转换.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/03Ear%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B-%E6%A8%A1%E5%9E%8B%E6%9D%83%E9%87%8D%E6%96%87%E4%BB%B6%E7%B2%BE%E7%AE%80%E8%BD%AC%E6%8D%A2.ipynb)
- [04Ear目标检测-预测.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/04Ear%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B-%E9%A2%84%E6%B5%8B.ipynb)
- [05Ear关键点检测-训练RTMPose.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/05Ear%E5%85%B3%E9%94%AE%E7%82%B9%E6%A3%80%E6%B5%8B-%E8%AE%AD%E7%BB%83RTMPose.ipynb)
- [06Ear关键点检测-可视化训练日志.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/06Ear%E5%85%B3%E9%94%AE%E7%82%B9%E6%A3%80%E6%B5%8B-%E5%8F%AF%E8%A7%86%E5%8C%96%E8%AE%AD%E7%BB%83%E6%97%A5%E5%BF%97.ipynb)
- [07Ear关键点检测预测-命令行.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/07Ear%E5%85%B3%E9%94%AE%E7%82%B9%E6%A3%80%E6%B5%8B%E9%A2%84%E6%B5%8B-%E5%91%BD%E4%BB%A4%E8%A1%8C.ipynb)
- [08Ear关键点检测预测-Python API.ipynb](https://github.com/chg0901/openmmlab2-hong/blob/main/Assignment1/08Ear%E5%85%B3%E9%94%AE%E7%82%B9%E6%A3%80%E6%B5%8B%E9%A2%84%E6%B5%8B-Python%20API.ipynb)




# 训练权重

RTMDet-tiny模型权重
[best_coco_bbox_mAP_epoch_196.pth](https://drive.google.com/file/d/1oBaGq98r5VySlMtCbLCavkStpmgxNLYl/view?usp=drive_link)

RTMPose-s模型权重
[best_PCK_epoch_255.pth](https://drive.google.com/file/d/1lFYuInSq_YISKDQp1PAhuaLUMRMxLsos/view?usp=sharing)

对应的模型轻量化转换权重

[best_coco_bbox_mAP_epoch_196_zip-dc2ee3bc.pth](https://drive.google.com/file/d/12NlnTy3P7-oVaMaYev3xr3tZbb2zFC6l/view?usp=sharing)

[best_PCK_epoch_255_zip-d1bf22ba.pth](https://drive.google.com/file/d/1WQRyWI22EJji4IwI9M_1ye_ljUwuAIyq/view?usp=sharing)

# 笔记链接
## OpenMMLab AI实战营 第二期 小结与心得
[OpenMMLab AI实战营 第二期 小结与心得](https://zhuanlan.zhihu.com/p/638331721)


## MMPose&RTMPose
[OpenMMLab 开源项目介绍 【OpenMMLab AI实战营 第二期 Day1】 - 知乎](https://zhuanlan.zhihu.com/p/633893792)

[人体姿态估计概论笔记【OpenMMLab AI实战营第二期Day2】 - 知乎](https://zhuanlan.zhihu.com/p/634214371)

[MMPose实战--基于耳朵穴位数据集【OpenMMLab AI实战营第二期Day3+作业1】 - 知乎](https://zhuanlan.zhihu.com/p/634511756)

[https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment1](https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment1)


## MMPretrain
[深度学习预训练与MMPretrain【OpenMMLab AI实战营 第二期 Day4】 - 知乎](https://zhuanlan.zhihu.com/p/634874025)

[MMPretrain实战--基于水果&猫狗数据集使用mim工具训练测试分析结果与含中文字符的可视化【OpenMMLab AI实战营第二期Day5+作业2】 - 知乎](https://zhuanlan.zhihu.com/p/635695648)

[MMPretrain 官方文档:学习配置文件与使用现有模型进行推理 - 知乎](https://zhuanlan.zhihu.com/p/636649634)

[https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment2](https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment2)



## MMDetection&RTMDet
[目标检测概述 【OpenMMLab AI实战营 第二期 Day6】 - 知乎](https://zhuanlan.zhihu.com/p/636775532)

[MMDetection代码课笔记 【OpenMMLab AI实战营 第二期 Day7】 - 知乎 - 知乎](https://zhuanlan.zhihu.com/p/636153265)

[MMDetection实战--基于RTMDet在Cats和10类饮料数据集上的目标检测与MMYoLo Grad-Based CAM可视化目标检测概述 【OpenMMLab AI实战营 第二期 作业2】 - 知乎 - 知乎](https://zhuanlan.zhihu.com/p/636155525)

[https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment3](https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment3)



## MMSegamentation
[语义分割概述 【OpenMMLab AI实战营 第二期 Day8】 - 知乎](https://zhuanlan.zhihu.com/p/637420070)

[语义分割实战--基于MMSegamentation的西瓜数据集上的语义分割【OpenMMLab AI实战营 第二期 Day9+作业4】 - 知乎](https://zhuanlan.zhihu.com/p/638323261)

[https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment4](https://github.com/chg0901/openmmlab2-hong/tree/main/Assignment4)



## MMagic
[图像超分辨率【MMagic理论基础】【OpenMMLab AI实战营 第二期 Day10】 - 知乎](https://zhuanlan.zhihu.com/p/637901406)

[MMagic实战上手【OpenMMLab AI实战营 第二期 Day11+作业5】 - 知乎](https://zhuanlan.zhihu.com/p/637901406)



## OpenMMLab算法库使用技巧
[OpenMMLab算法库使用技巧](https://zhuanlan.zhihu.com/p/634877254)

还有的是我对这次课程里遇到的常见问题的总结，后续会添加更多更新，来帮助正在学习的同学们。



