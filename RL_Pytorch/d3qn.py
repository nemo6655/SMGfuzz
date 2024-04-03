# import gym
import collections
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from sklearn.decomposition import KernelPCA
#np.set_printoptions(threshold=np.inf)
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
# Hyperparameters 超参数
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
MAX_EPISODE = 100

#考虑到对一个新的全局bitmap,前几次（普通）变异也会快速覆盖很多边，而后续（哪怕是精彩的）变异也只会新增少量的边，符合指数函数规律，需要加权重来计算奖励。
#参数由find_eps_reward_law.py得到
n_epi_a = 1/2.06457084e+03
n_epi_b = 3.10658261e-02
#文件夹路径
seed_path = 'Decode_Data/seed_vec'
bitmap_path = 'Decode_Data/bitmap'
statemap_path = 'Decode_Data/statemap'

#获取seed\bitmap\statemap列表
seeds = os.listdir(seed_path)
bitmaps = os.listdir(bitmap_path)
statemaps = os.listdir(statemap_path)

torch.manual_seed(1)

n_features = 258 # 状态空间的大小，种子文件序号（1）+该文件可触发的状态数（1）+独热码（16*16）
n_actions = 256  #动作空间的大小，不是维度，是card()，表示随机翻转几个,1维，取值范围是1-256

#获取列表的长度，bitmap或statemap都可
choice_num = len(bitmaps)

#储存最大的reward，以及此时的state
best_predict = 0
best_epi = 0
best_state = np.zeros(n_features)

#维护一个全局的bitmap,记录全部被覆盖的边，只要这个图有增加，就给reward
total_bits = np.zeros((n_actions,n_actions))

top_k_list = []#字典列表，储存前K大的reward对应的seed（打印）/state（打印）/reward（排序）

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

def save_to_list(epi, reward, best_seed, best_state):
    global top_k_list
    new_dict = {
        "n_epi": epi,
        "reward": reward,
        "best_seed": best_seed,
        "best_state": best_state
    }
    top_k_list.append(new_dict)

def add_noise(matrix, mean, std):
    noise = np.random.normal(mean, std, size=matrix.shape)
    noisy_matrix = np.clip(matrix + noise, 0, 1)
    return noisy_matrix.astype(int)

def find_statemap_and_avaistate(state0):
    choice_statemap = statemaps[int(state0)]
    choice_seed = [string for string in seeds if len(string) >= 10 and string[3:9] == choice_statemap[9:15]]
    # 使用 np.loadtxt 读取文件
    matrix = np.loadtxt(statemap_path+'/'+choice_statemap, delimiter=' ', dtype=int)
    # 将矩阵 reshape 成 1*256
    reshaped_matrix = matrix.reshape(1, 256)
    # 判断位置是否有 1
    ones_indices = np.where(reshaped_matrix == 1)[1]
    return choice_statemap, choice_seed, ones_indices, len(ones_indices)

def set_top_n_to_1(matrix, n):
    # 找到前N个最大的数的位置
    indices = np.argpartition(matrix.flatten(), -n)[-n:]
    # 创建全零矩阵
    result = np.zeros_like(matrix)
    # 将前N个最大数的位置置1
    result.flat[indices] = 1
    return result

def add_and_threshold(matrix1, matrix2):
    # 将两个矩阵相加
    result = matrix1 + matrix2
    # 将大于1的元素置1，其余元素保持不变
    result = np.where(result > 1, 1, result)
    return result

def get_nearest_file(directory):
    now = datetime.datetime.now()
    nearest_file = None
    nearest_time_diff = None
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_time = datetime.datetime.fromtimestamp(os.path.getctime(file_path))
            time_diff = abs(now - file_time)
            if nearest_time_diff is None or time_diff < nearest_time_diff:
                nearest_file = file_path
                nearest_time_diff = time_diff
    return nearest_file

# 自行编写的强化学习环境，用映射代替一次fuzz
class Fuzzenv():
    def __init__(self):
        self.state_space_size = n_features
        self.action_space_size = n_actions
        self.current_state = None

    def reset(self):
        # 初始化当前状态
        #只有每一次初始化才能改变种子文件，如果种子文件不好，就只能重开等下一次初始化
        # state[0]代表随机选择一个选取种子文件
        # state[1]代表该种子文件总共有可触发的状态数量
        # state[2:]代表随机选择若干个该种子文件可能触发的状态

        vector = np.zeros(n_features)  # 初始化全0的向量
        state0 = random.randint(0, choice_num-2)
        #根据reset随机生成的state1，找到seed列表、statemap列表里所选择的seed、state，可能触发状态的位置，总共有可触发的状态数量

        choice_statemap, choice_seed, ones_indices, state_num = find_statemap_and_avaistate(state0)

        vector[0] = state0
        vector[1] = state_num
        #
        #random_count = np.random.randint(1, state_num-1)  # 生成1到state_num之间的随机个数
        #random_indices = np.random.choice(range(2, state_num+2), random_count, replace=False)  # 随机选择索引位置
        vector[2] = 1  # 将选中的索引位置设置为1

        self.current_state = vector
        return self.current_state

    def step(self, action,n_epi):
        global best_predict
        global best_state
        global total_bits
        global best_epi
        # 执行动作并返回下一个状态和奖励
        # 动作就是生成若干个可触发的状态
        count =0
        state1 = self.current_state[0]

        choice_statemap, choice_seed, ones_indices, state_num = find_statemap_and_avaistate(state1)

        #随机生成新状态，在可能触发的位置里，随机选择若干个状态，state_num是全部可能触发状态的数量
        next_state = self.current_state
        random_count = random.randint(1, action)  # 生成1到state_num之间的随机个数
        random_indices = np.random.choice(range(2, state_num+2), random_count, replace=False)  # 随机选择索引位置
        next_state[random_indices] = 1 - next_state[random_indices] # 将选中的索引位置的值翻转

        #预测旧状态可能影响的bitmap和新状态可能影响的bitmap
        predict = self.pred_bitmap(choice_seed,self.current_state) 
        next_predict = self.pred_bitmap(choice_seed,next_state)

        #计算奖励
        reward = n_epi_a*np.exp(n_epi_b*n_epi)*self.calculate_reward(next_predict)

        #取state_num作为一个参考值，变异次数超过这个值，就认为是种子不行，done=True，重开，下一把
        #如果有所提升，就带着奖励继续
        while count < state_num:
            if reward > 0:
                if best_predict < reward and n_epi>1:
                    if best_state[0]==next_state[0]:
                        best_predict += reward
                    else:
                        best_predict = reward
                    best_state = next_state
                    best_epi = n_epi
                    
                self.current_state = next_state
                total_bits = add_and_threshold(total_bits, next_predict)
                return self.current_state, reward, False

            else:
                random_count = random.randint(1, action)  # 生成1到state_num之间的随机个数
                random_indices = np.random.choice(range(2, state_num+2), random_count, replace=False)  # 随机选择索引位置
                next_state[random_indices] = 1 - next_state[random_indices] # 将选中的索引位置的值翻转
                next_predict = self.pred_bitmap(choice_seed,next_state)
                reward = self.calculate_reward(next_predict)
                count += 1

        self.current_state = next_state
        return self.current_state, reward, True

    def calculate_reward(self,next_predict):
        global total_bits
        new_bits = np.where((total_bits==0) & (next_predict==1),1,0)

        return sum(sum(new_bits))

    def pred_bitmap(self, choice_seed, state):
        kpca_state = KernelPCA(n_components=16, kernel='rbf')
        choice_seed_data = np.loadtxt(seed_path+'/'+choice_seed[0])
        choice_seed_data /= np.max(choice_seed_data)
        state_map_data = state[2:].reshape(16,16)
        state_map_data = add_noise(state_map_data, mean=0, std=0.01)
        state_map_data = kpca_state.fit_transform(state_map_data)
        input = (torch.tensor(state_map_data).unsqueeze(0).to(torch.float32),torch.tensor(choice_seed_data).to(torch.float32))
        outputs = net(*input)
        predict_bitmap = set_top_n_to_1(sum(outputs.detach().numpy()),int(sum(sum(sum(outputs.detach().numpy())))))
        return predict_bitmap

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_next_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_next_lst.append(s_)
            done_mask_lst.append([done_mask])

        return torch.tensor(np.array(s_lst), dtype=torch.float), torch.tensor(np.array(a_lst)), \
               torch.tensor(np.array(r_lst)), torch.tensor(np.array(s_next_lst), dtype=torch.float), \
               torch.tensor(np.array(done_mask_lst))

    def size(self):
        return len(self.buffer)

class DQNDuelingNet(nn.Module):
    def __init__(self):
        super(DQNDuelingNet, self).__init__()
        hidden_dims = 128
        self.feature_layer = nn.Sequential(nn.Linear(n_features, hidden_dims),
                                           nn.ReLU())
        self.value_layer = nn.Linear(hidden_dims, 1)
        self.advantage_layer = nn.Linear(hidden_dims, n_actions)

    def forward(self, x):
        feature = self.feature_layer(x)
        value = self.value_layer(feature)
        advantage = self.advantage_layer(feature)

        avg_advantage = torch.mean(input=advantage, dim=-1, keepdim=True)
        q_values = value + (advantage - avg_advantage)
        return q_values

class Dueling_DQN:
    def __init__(self):
        # [target_net, evaluate_net]
        self.evaluate_net = DQNDuelingNet()
        self.target_net = type(self.evaluate_net)()
        self.target_net.load_state_dict(self.evaluate_net.state_dict())  # copy weights and stuff

        self.optimizer = torch.optim.Adam(self.evaluate_net.parameters(),
                                          learning_rate)
        self.memory = ReplayBuffer()

    def train(self):
        s, a, r, s_, done_mask = self.memory.sample(batch_size)

        q_out = self.evaluate_net(s)
        q_a = q_out.gather(1, a)

        # 与Dueling DQN的不同之处
        # max_q_prime = torch.max(self.target_net(s_), dim=1, keepdim=True).values
        #  target = r + gamma * max_q_prime * done_mask
        q_target_next = self.target_net(s_).detach()
        q_eval_next = self.evaluate_net(s_).detach()
        q_next = q_target_next.gather(1, q_eval_next.argmax(axis=1).reshape(-1, 1))
        target = r + gamma * q_next * done_mask

        loss = F.smooth_l1_loss(q_a, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sample_action(self, obs, epsilon):
        coin = random.random()
        #如果是随机选择动作
        if coin < epsilon:
            random_action = np.random.randint(1, int(obs[1]))  # 生成1到state_num之间的随机个数,randint是左闭右闭的！！！
            return random_action
        else: #按照网络的估计来选择动作
            out = self.evaluate_net(obs)
            return out.argmax().item() % int(obs[1])+1

#创建以神经网络代替小游戏的强化学习环境，实例化
env = Fuzzenv()
trainer = Dueling_DQN()
#创建网络，并且读取训练好的数据
net = MyNet()

directory = 'Train_Result/model'  # 替换为实际的目录路径
lastest_model = get_nearest_file(directory)
net.load_state_dict(torch.load(lastest_model))

print_interval = 1
score = 0.0

for n_epi in range(MAX_EPISODE):
    epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
    s = env.reset()
    done = False
    need_save = False

    while not done:
        a = trainer.sample_action(torch.from_numpy(s).float(), epsilon)
        s_, r, done = env.step(a,n_epi)
        done_mask = 0.0 if done else 1.0
        trainer.memory.put((s, a, r / 100.0, s_, done_mask))
        s = s_
        if n_epi>1 and score < sum(sum(total_bits)):#判断这个step后，total_bits有没有增加，增加则需要记录
            need_save = True
        score = sum(sum(total_bits))
        if done:
            break
        
    if need_save: 
        save_to_list(n_epi,best_predict,seeds[int(best_state[0])],best_state[2:])

    if trainer.memory.size() > 2000:
        trainer.train()

    if n_epi % print_interval == 0 and n_epi != 0:
        trainer.target_net.load_state_dict(trainer.evaluate_net.state_dict())
        print("n_episode :{}, score : {:.2f}, n_buffer : {}, eps : {:.2f}%".format(
            n_epi, score / print_interval, trainer.memory.size(), epsilon * 100))
        #score = 0.0
print('------Reinforcement Learning Preferred Outcomes------')
print("best predict's reward:"+str(best_predict))
print("best state:"+str(best_state))
print('total_bits:'+str(sum(sum(total_bits)))+'/65536')
print('best epi:'+str(best_epi))
print('best seed:'+seeds[int(best_state[0])])
# 按照键值为数字的键进行排序
sorted_list = sorted(top_k_list, key=lambda x: x["reward"])
if best_predict==0:
    os.remove(lastest_model)
# 将每个字典写入txt文件
for i, d in enumerate(sorted_list):
    file_name = f"Train_Result/RL_Result/"+d["best_seed"][3:9]+'_'+"{:.4f}".format(d["reward"])
    state_list = d["best_state"].astype(int)
    for j in range(0,len(state_list)):
        if state_list[j]==1:
            file_name += '_'+str(j).zfill(3)
    with open(file_name, "w") as f:
        np.savetxt(f, d["best_state"].astype(int), fmt='%d')