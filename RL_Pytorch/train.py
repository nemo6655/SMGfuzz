import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import sys
import random

#torch.set_printoptions(threshold=sys.maxsize)
# 设置文件夹的路径
folder1_path = "Seed_Vec/"
folder2_path = "Decode_Data/statemap/"
folder3_path = "Decode_Data/bitmap/"
# 定义网络架构
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU()
        )
        self.decnn = nn.Sequential(
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1,output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = self.cnn(x1)
        x1 = x1.view(-1)
        x2 = self.fc1(x2)
        '''
        scale_factor = x2.mean() / x1.mean()
        if scale_factor >10 and scale_factor <100000:
            x = torch.cat((x1*scale_factor, x2), dim=0)
        else:
            x = torch.cat((x1, x2), dim=0)'''
        x = torch.cat((x1, x2), dim=0)
        x = self.fc2(x)
        x = torch.reshape(x,(32,32)).unsqueeze(0)
        x = self.decnn(x)
        return x

def split_dataset(data, train_ratio):
    random.shuffle(data)  # 随机打乱列表顺序
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

# 创建网络实例
net = MyNet()

# 定义损失函数和优化器
criterion = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.05)

num_epochs=50

train_loader = []
# 获取文件夹1中的所有txt文件
folder1_files = [f for f in os.listdir(folder1_path) if f.endswith(".txt")]
# 获取文件夹2中的所有txt文件
folder2_files = [f for f in os.listdir(folder2_path) if f.endswith(".txt")]
# 获取文件夹3中的所有txt文件
folder3_files = [f for f in os.listdir(folder3_path) if f.endswith(".txt")]

for file2 in folder2_files:
    file2_name = file2[9:15]
    for file1 in folder1_files:
        file1_name = file1[4:10]
        # 判断两个文件名是否相同
        if file1_name == file2_name:
            # 读取文件夹1中的列向量数据
            file1_data = np.loadtxt(os.path.join(folder1_path, file1))
            
            # 读取文件夹2中的矩阵数据
            file2_data = np.loadtxt(os.path.join(folder2_path, file2))
            file3_data = np.loadtxt(os.path.join(folder3_path, 'bitmap,'+file2_name+'.txt'))
            
            train_loader.append((file1_data,file2_data,file3_data))

train_data, test_data = split_dataset(train_loader, 0.7)
print(str(len(train_data))+"+"+str(len(test_data)))
# 训练网络
for epoch in range(num_epochs):
    running_loss = 0.0
    testing_loss = 0.0
    for i, (seed, statemap, bitmap) in enumerate(train_data):
        optimizer.zero_grad()
        scale_factor = statemap.mean() / seed.mean()
        if scale_factor >10 and scale_factor <100000:
            seed = seed * scale_factor
        input = (torch.tensor(statemap).unsqueeze(0).to(torch.float32),torch.tensor(seed).to(torch.float32))
        outputs = net(*input)
        bitmap_tensor = torch.tensor(bitmap, dtype=torch.float32).unsqueeze(0)
        loss = criterion(outputs, bitmap_tensor)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    for j, (seed_t, statemap_t, bitmap_t) in enumerate(test_data):
        scale_factor = statemap_t.mean() / seed_t.mean()
        if scale_factor >10 and scale_factor <100000:
            seed_t = seed_t * scale_factor
        input_t = (torch.tensor(statemap_t).unsqueeze(0).to(torch.float32),torch.tensor(seed_t).to(torch.float32))
        outputs_t = net(*input_t)
        bitmap_tensor_t = torch.tensor(bitmap, dtype=torch.float32).unsqueeze(0)
        loss_t = criterion(outputs_t, bitmap_tensor_t)
        testing_loss += loss_t.item()
    
    print('Epoch [{}/{}], TrainLoss: {:.5f}, TestLoss: {:.5f}'.format(epoch+1, num_epochs, running_loss / len(train_data),testing_loss / len(test_data)))




