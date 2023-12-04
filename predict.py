import torch
from Resnet34 import Resnet
from PIL import Image
from torchvision import transforms

import matplotlib.pyplot as plt
import json 


# 需要使用和验证(validate)时相同的transform
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载图片
img = Image.open(r"flower_photos\prediction\向日葵1.jpg")
plt.show(img)

# transform
img = data_transform(img)

# 扩展batch_size维度
img = torch.unsqueeze(img,0)

# 加载分类集class_idx
class_idx = None
with open("class_idx.json",'r') as f:
    class_idx = json.load(fp=f)

# create model
resnet34 = Resnet()
resnet34.layer5[2] = torch.nn.Linear(512,5)

# 加载权重
weight_path = 'fine_tuning_resnet34.pth'
resnet34.load_state_dict(torch.load(weight_path))

# Validate
resnet34.eval()

with torch.no_grad():
    output = resnet34(img)
    print(output.shape)
    prediction = torch.squeeze(output) # 将维度为1的维度去掉 应该只有一维了
    prediction_index = torch.argmax(prediction)

print(f"分类结果:{class_idx[str(prediction_index)]}")