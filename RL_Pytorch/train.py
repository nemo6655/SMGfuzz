import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.decomposition import KernelPCA
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#torch.set_printoptions(threshold=sys.maxsize) #使print显示完全
# 获取当前时间
current_time = datetime.now()
# 将当前时间格式化为字符串
time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

# 绘制loss损失函数（保存文件名包含时间）
def plot_loss(train_loss, test_loss):
    x = range(1, len(train_loss) + 1)
    plt.plot(x, train_loss, label='Train Loss')
    plt.plot(x, test_loss, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('Train_Result/fig/loss'+time_str+'.svg', format='svg')

#添加高斯白噪声的函数   
def add_noise(matrix, mean, std):
    noise = np.random.normal(mean, std, size=matrix.shape)
    noisy_matrix = np.clip(matrix + noise, 0, 1)
    return noisy_matrix.astype(int)

# 设置文件夹的路径
folder1_path = "Decode_Data/seed_vec/"
folder2_path = "Decode_Data/statemap/"
folder3_path = "Decode_Data/bitmap/"

# 定义网络架构
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1024),
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
net.to(device)
# 定义损失函数和优化器
criterion = nn.BCELoss().to(device)
optimizer = optim.Adam(net.parameters(), lr=0.001)

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
            #file1_data /=np.max(file1_data)
            # 读取文件夹2中的矩阵数据
            file2_data = np.loadtxt(os.path.join(folder2_path, file2))
            file2_data = add_noise(file2_data, mean=0, std=0.01) #加噪声
            file3_data = np.loadtxt(os.path.join(folder3_path, 'bitmap,'+file2_name+'.txt'))
            #file3_data /=np.max(file3_data)
            file3_data = np.where(file3_data != 0, 1, file3_data)#大于1的位置置1
            train_loader.append((file1_data,file2_data,file3_data))


train_data, test_data = split_dataset(train_loader, 0.8) #train:test=9:1

print("train:"+str(len(train_data))+",test:"+str(len(test_data)))

# 训练网络
train_loss = []
test_loss = []

for epoch in range(num_epochs):
    running_loss = 0.0
    testing_loss = 0.0
    for i, (seed, statemap, bitmap) in enumerate(train_data):
        optimizer.zero_grad()
        # 创建KernelPCA对象
        kpca_state = KernelPCA(n_components=16, kernel='rbf')
        # 进行核方法的PCA降维
        statemap = kpca_state.fit_transform(statemap)

        input = (torch.tensor(statemap).unsqueeze(0).to(torch.float32).to(device),torch.tensor(seed).to(torch.float32).to(device))
        outputs = net(*input)
        #print(sum(sum(sum(outputs))))
        #print(np.mean(np.mean(bitmap)))
        bitmap_tensor = torch.tensor(bitmap, dtype=torch.float32).unsqueeze(0).to(device)
        loss = criterion(outputs, bitmap_tensor)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    for j, (seed_t, statemap_t, bitmap_t) in enumerate(test_data):
        # 进行核方法的PCA降维
        statemap_t = kpca_state.fit_transform(statemap_t)

        input_t = (torch.tensor(statemap_t).unsqueeze(0).to(torch.float32).to(device),torch.tensor(seed_t).to(torch.float32).to(device))
        outputs_t = net(*input_t)
        bitmap_tensor_t = torch.tensor(bitmap, dtype=torch.float32).unsqueeze(0).to(device)
        loss_t = criterion(outputs_t, bitmap_tensor_t)
        testing_loss += loss_t.item()

    train_loss.append(running_loss / len(train_data))
    test_loss.append(testing_loss / len(test_data))
    print('Epoch [{}/{}], TrainLoss: {:.5f}, TestLoss: {:.5f}'.format(epoch+1, num_epochs, running_loss / len(train_data),testing_loss / len(test_data)))

#保存参数
torch.save(net.state_dict(), 'Train_Result/model/model_'+time_str+'.pth')
#绘制并保存图像
#plot_loss(train_loss, test_loss)



