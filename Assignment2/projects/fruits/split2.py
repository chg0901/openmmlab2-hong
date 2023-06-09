import os
import shutil
import random

# 定义数据集路径和保存路径
data_dir = 'fruit30_train'
new_data_dir = 'fruit30_dataset'
os.makedirs(new_data_dir, exist_ok=True)

train_dir = os.path.join(new_data_dir, 'training_set')
val_dir = os.path.join(new_data_dir, 'val_set')

# 定义训练集和验证集的比例
train_ratio = 0.7
val_ratio = 0.3

# 创建保存路径
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# 获取数据集中所有子文件夹的名称
subdirs = os.listdir(data_dir)


# 遍历所有子文件夹
for subdir in subdirs:
    subdir_path = os.path.join(data_dir, subdir)
    if os.path.isdir(subdir_path):
        # 获取当前子文件夹中所有图像的文件名
        images = os.listdir(subdir_path)
        # 随机打乱图像的顺序
        random.shuffle(images)
        # 计算训练集和验证集的大小
        num_images = len(images)
        num_train = int(train_ratio * num_images)
        num_val = int(val_ratio * num_images)
        # 将图像按比例划分为训练集和验证集
        train_images = images[:num_train]
        val_images = images[num_train:num_train+num_val]
        # 将划分好的训练集和验证集保存到对应的文件夹中
        for image in train_images:
            src_path = os.path.join(subdir_path, image)
            dst_path = os.path.join(train_dir, subdir, image)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
        for image in val_images:
            src_path = os.path.join(subdir_path, image)
            dst_path = os.path.join(val_dir, subdir, image)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy(src_path, dst_path)
print('complete！！！')