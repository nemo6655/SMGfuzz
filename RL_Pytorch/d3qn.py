# import gym
import collections
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
from train import MyNet, add_noise
import torch.nn.functional as F
import torch.optim as optim
import os
from sklearn.decomposition import KernelPCA

# Hyperparameters 超参数
learning_rate = 0.0005
gamma = 0.98
buffer_limit = 50000
batch_size = 32
MAX_EPISODE = 1000
RENDER = False

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

def find_statemap_and_avaistate(state0):
    choice_statemap = statemaps[int(state0)]
    choice_seed = [string for string in seeds if len(string) >= 10 and string[4:10] == choice_statemap[9:15]]
    # 使用 np.loadtxt 读取文件
    matrix = np.loadtxt(statemap_path+'/'+choice_statemap, delimiter=' ', dtype=int)
    # 将矩阵 reshape 成 1*256
    reshaped_matrix = matrix.reshape(1, 256)
    # 判断位置是否有 1
    ones_indices = np.where(reshaped_matrix == 1)[1]
    return choice_statemap, choice_seed, ones_indices, sum(sum(matrix))

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
        #在可能触发的位置里，随机选择若干个状态，state_num是全部可能触发状态的数量
        random_count = np.random.randint(1, state_num-1)  # 生成1到state_num之间的随机个数
        random_indices = np.random.choice(range(2, state_num+2), random_count, replace=False)  # 随机选择索引位置
        vector[random_indices] = 1  # 将选中的索引位置设置为1

        self.current_state = vector
        return self.current_state

    def step(self, action):
        # 执行动作并返回下一个状态和奖励
        # 动作就是生成若干个可触发的状态
        count =0
        state1 = self.current_state[0]

        choice_statemap, choice_seed, ones_indices, state_num = find_statemap_and_avaistate(state1)

        #随机生成不重复的新状态
        next_state = self.current_state
        random_count = random.randint(1, action)  # 生成1到state_num之间的随机个数
        random_indices = np.random.choice(range(2, state_num+2), random_count, replace=False)  # 随机选择索引位置
        next_state[random_indices] = 1 - next_state[random_indices] # 将选中的索引位置的值翻转

        #预测旧状态可能影响的bitmap和新状态可能影响的bitmap
        predict = self.pred_bitmap(choice_seed,self.current_state) 
        next_predict = self.pred_bitmap(choice_seed,next_state)

        #计算奖励
        reward = self.calculate_reward(predict, next_predict)

        #取state_num作为一个参考值，变异次数超过这个值，就认为是种子不行，done=True，重开，下一把
        #如果有所提升，就带着奖励继续
        while count < state_num:
            if reward > 0:
                self.current_state = next_state
                return self.current_state, reward, False
            else:
                random_count = random.randint(1, action)  # 生成1到state_num之间的随机个数
                random_indices = np.random.choice(range(2, state_num+2), random_count, replace=False)  # 随机选择索引位置
                next_state[random_indices] = 1 - next_state[random_indices] # 将选中的索引位置的值翻转
                next_predict = self.pred_bitmap(choice_seed,next_state)
                reward = self.calculate_reward(predict, next_predict)
                count += 1

        self.current_state = next_state
        return self.current_state, reward, True

    def calculate_reward(self, predict,next_predict):
        return np.clip(next_predict-predict,-1,1)

    def pred_bitmap(self, choice_seed, state):
        kpca_state = KernelPCA(n_components=16, kernel='rbf')
        choice_seed_data = np.loadtxt(seed_path+'/'+choice_seed[0])
        state_map_data = state[2:].reshape(16,16)
        state_map_data = add_noise(state_map_data, mean=0, std=0.01)
        state_map_data = kpca_state.fit_transform(state_map_data)
        input = (torch.tensor(state_map_data).unsqueeze(0).to(torch.float32),torch.tensor(choice_seed_data).to(torch.float32))
        outputs = net(*input)
        total_edge = sum(sum(sum(outputs.detach().numpy())))
        #print(total_edge)
        return total_edge

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

print_interval = 20
score = 0.0

for n_epi in range(MAX_EPISODE):
    epsilon = max(0.01, 0.08 - 0.01 * (n_epi / 200))  # Linear annealing from 8% to 1%
    s = env.reset()
    done = False

    while not done:
        a = trainer.sample_action(torch.from_numpy(s).float(), epsilon)
        s_, r, done = env.step(a)
        done_mask = 0.0 if done else 1.0
        trainer.memory.put((s, a, r / 100.0, s_, done_mask))
        s = s_
        score += r
        if done:
            break

    if trainer.memory.size() > 2000:
        trainer.train()

    if n_epi % print_interval == 0 and n_epi != 0:
        trainer.target_net.load_state_dict(trainer.evaluate_net.state_dict())
        print("n_episode :{}, score : {:.6f}, n_buffer : {}, eps : {:.2f}%".format(
            n_epi, score / print_interval, trainer.memory.size(), epsilon * 100))
        score = 0.0
