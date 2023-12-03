import os
import random
import shutil

def split_dataset(source_folder, train_folder, test_folder, split_ratio=0.8):
    """
    将源文件夹中的图片按指定比例划分为训练集和测试集，并保存到目标文件夹中。

    参数:
    - source_folder: 包含所有图片的源文件夹路径
    - train_folder: 保存训练集图片的目标文件夹路径
    - test_folder: 保存测试集图片的目标文件夹路径
    - split_ratio: 训练集所占比例,默认为0.8(即80%训练集,20%测试集)
    """
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    for class_folder in os.listdir(source_folder):
        class_path = os.path.join(source_folder, class_folder)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if f.endswith('.jpg') or f.endswith('.png')]
            random.shuffle(images)

            split_index = int(len(images) * split_ratio)
            train_images = images[:split_index]
            test_images = images[split_index:]

            for img in train_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(train_folder, class_folder, img)
                if not os.path.exists(os.path.join(train_folder, class_folder)):
                    os.makedirs(os.path.join(train_folder, class_folder))
                shutil.copy(src_path, dst_path)

            for img in test_images:
                src_path = os.path.join(class_path, img)
                dst_path = os.path.join(test_folder, class_folder, img)
                if not os.path.exists(os.path.join(test_folder, class_folder)):
                    os.makedirs(os.path.join(test_folder, class_folder))
                shutil.copy(src_path, dst_path)

# 示例用法
source_folder = 'flower_photos/data'  # 替换为包含五个文件夹的文件夹路径
train_folder = 'flower_photos/train'
test_folder = 'flower_photos/test'

split_dataset(source_folder, train_folder, test_folder)
