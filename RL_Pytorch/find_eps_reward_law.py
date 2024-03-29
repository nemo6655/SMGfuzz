import random
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def generate_points(num_points):
    points = set()
    while len(points) < num_points:
        x = random.randint(0, 255)
        y = random.randint(0, 255)
        points.add((x, y))
    return list(points)

# 定义不同的拟合函数
def exponential_function(x, a, b):
    return a * np.exp(-b * x)


def simulate_game(num_iterations, num_points):
    global_map = [[0] * 256 for _ in range(256)]  # 初始化全局地图
    new_points_counts = []  # 记录每轮新占领点数
    for i in range(0,num_iterations):
        points = generate_points(num_points)
        new_points = 0  # 统计新占领点数
        for point in points:
            x, y = point
            if global_map[x][y] == 0:
                new_points += 1
            global_map[x][y] += 1  # 在全局地图上标记点的占领次数
        new_points_counts.append(new_points)  # 记录新占领点数

    # 绘制变化曲线
    averaged_vector = np.mean(np.array_split(new_points_counts, len(new_points_counts)//20), axis=1)
    plt.bar(range(10, num_iterations+10,20), averaged_vector, width=15,color=(155/255,184/255,205/255), label='Monte Carlo Simulation')

    popt_exponential, pcov_exponential = curve_fit(exponential_function, range(1, num_iterations+1), new_points_counts)
    plt.plot(range(1, num_iterations+1), exponential_function(range(1, num_iterations+1), *popt_exponential), '-', color=(238/255,199/255,89/255),linewidth=2,label='Fit Curve: $y=ae^{-bx}$')
    #plt.plot(range(1,num_iterations+1),exponential_function(range(1,num_iterations+1), 1,-popt_exponential[1]),'-',linewidth=2,label='$f_{ offset }$: $y=ae^{-bx}$')
    plt.xlabel('Iteration')
    plt.ylabel('Newly added edges in bitmap')
    plt.legend()
    # 增加横线grid
    plt.grid(axis='y', linestyle='--')

    # 设置背景颜色为浅灰色
    plt.tight_layout()#调整整体空白

    # 保存成svg格式
    plt.savefig('Train_Result/fig/eps_reward_law.pdf', format='pdf')
    plt.show()
    print(popt_exponential)

num_iterations = 400  # 迭代轮次
num_points = 1000  # 每轮生成的点数
simulate_game(num_iterations, num_points)



