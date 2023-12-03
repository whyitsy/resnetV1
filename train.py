import torchvision
from torchvision import transforms
import torch
import json
import tqdm

import Resnet34


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("当前使用设备:",device)

# 定义transform
data_transform = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(224), # 图像增广
        transforms.RandomHorizontalFlip(), # 图像增广
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize(256), # 长宽比固定,最小边缩放到256
        transforms.CenterCrop(224), # 中心裁剪是很常用的增广方法
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 这里是把图片加载到内存并标号
train_image_path = 'flower_photos/train'
train_dataset = torchvision.datasets.ImageFolder(train_image_path,transform=data_transform["train"])
train_num = len(train_dataset)

test_image_path = 'flower_photos/test'
test_dataset = torchvision.datasets.ImageFolder(test_image_path,transform=data_transform['test'])
test_num = len(test_dataset)

# 获取类别
# class_dict = test_dataset.classes # 类别list
# print(class_dict)
# print(test_dataset.find_classes('flower_photos/train')) # 第一个同test_dataset.classes,第二个是类别和index的字典
# {'daisy': 0, 'dandelion': 1, 'roses': 2, 'sunflowers': 3, 'tulips': 4}
class_dict = test_dataset.find_classes('flower_photos/train')[1]
class_dict = {value: key for key, value in class_dict.items()}

# 存入json file
# json.dumps()用于将数据转换为JSON格式的字符串，而json.dump()用于将数据写入打开的文件
with open('class_idx.json','w') as json_file:
    json.dump(class_dict,json_file,indent=4)

batch_size = 32

# 加载迭代器
train_iter = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=0)
test_iter = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=0)

resnet34 = Resnet34.Resnet()
# print(resnet34.layer5[2])

resnet34.to(device)

pretrained_model_weights = 'resnet34-pre.pth'
resnet34.load_state_dict(torch.load(pretrained_model_weights),strict=False)

'''
查看网络结构是如何定义出来的
在网络构造函数中添加num_classes便于在prediction时直接调用
不用每次使用该网络时都要这样显式的改变类别
'''
# 加载预训练权重后修改最后全连接层的输出类别
resnet34.layer5[2] = torch.nn.Linear(in_features=512,out_features=5)

# 损失函数
loss_fn = torch.nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(resnet34.parameters(),lr=0.0001)

best_acc = 0.0
save_path = './fine_tuning_resnet34.pth'
epochs = 5

for epoch in range(epochs):
    # train
    resnet34.train()
    batch_loss = 0
    # 将train_iter进行迭代,返回步数便于统计,就不需要自己定义轮数了
    for step,data in tqdm.tqdm(enumerate(train_iter)):
        images,labels = data  # 这时一个batch
        optimizer.zero_grad() # 训练前清空梯度
        
        # 数据移动到cuda
        images = images.to(device)
        labels = labels.to(device)

        # 放入训练
        outputs = resnet34(images)

        # 计算损失
        loss = loss_fn(outputs,labels)

        # 反向传播计算梯度
        loss.backward()

        # 利用计算的梯度更新权重
        optimizer.step()

        # 累计损失
        batch_loss += loss.item()

    # eval
    resnet34.eval()

    acc = 0.0
    with torch.no_grad():
        for step,data in tqdm.tqdm(enumerate(test_iter)):
            images, labels = data

            images = images.to(device)
            labels = labels.to(device)

            outputs = resnet34(images)

            acc += (outputs.argmax(axis = 1) == labels).sum().item()

    # 测试集的准确率       
    acc = acc/len(images)

    # 保存准确率高的模型
    if best_acc < acc:
        best_acc = acc
        torch.save(resnet34.state_dict(),save_path)

    print(f"epoch:{epoch+1}, train_loss:{loss},validate_acc:{best_acc}")

print("Finished Training!")
    
    



        









