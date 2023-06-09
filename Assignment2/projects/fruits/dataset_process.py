import os

# 获取数据集文件夹路径
CustomDatasetPath = r'../../data/fruit'
# 获取数据集文件夹下的所有文件
CustomDatasetFile = os.listdir(CustomDatasetPath)
# 如果文件夹中不存在train、val、test文件夹，则创建
dataset_type = ['train', 'val', 'test']
for type in dataset_type:
    if type not in CustomDatasetFile:
        os.mkdir(os.path.join(CustomDatasetPath, type))
    else:
        # 清空文件夹
        os.removedirs(os.path.join(CustomDatasetPath, type))

# 遍历所有文件
for fruit_name in CustomDatasetFile:
    for type in dataset_type:
        os.mkdir(os.path.join(CustomDatasetPath, type, fruit_name))
    # 水果文件夹路径
    fruit_path = os.path.join(CustomDatasetPath, fruit_name)
    # 获取水果文件夹下的所有文件
    fruit_file = os.listdir(fruit_path)
    # 将水果文件夹下的所有文件分为训练集、验证集、测试集
    train_file = fruit_file[:int(len(fruit_file)*0.8)]
    val_file = fruit_file[int(len(fruit_file)*0.8):int(len(fruit_file)*0.9)]
    test_file = fruit_file[int(len(fruit_file)*0.9):]
    # 将训练集、验证集、测试集分别放入对应文件夹
    for file in train_file:
        os.rename(os.path.join(fruit_path, file), os.path.join(CustomDatasetPath, 'train', fruit_name, file))
    for file in val_file:
        os.rename(os.path.join(fruit_path, file), os.path.join(CustomDatasetPath, 'val', fruit_name, file))
    for file in test_file:
        os.rename(os.path.join(fruit_path, file), os.path.join(CustomDatasetPath, 'test', fruit_name, file))
    # 删除空文件夹
    os.removedirs(fruit_path)

