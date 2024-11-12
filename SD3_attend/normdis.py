import torch
import numpy as np
import pywt
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
def process_file(path, stats):
    data = torch.load(path)
    if not isinstance(data, list):
        print("加载的数据不是一个列表。")
        return

    # 初始化存储分量的数组
    LL_array, LH_array, HL_array, HH_array = [], [], [], []

    for item in data:
        c, h, w = item.shape
        temp_LL = np.zeros((c, h//2, w//2))
        temp_LH = np.zeros((c, h//2, w//2))
        temp_HL = np.zeros((c, h//2, w//2))
        temp_HH = np.zeros((c, h//2, w//2))
        
        for i in range(c):
            coeffs2 = pywt.dwt2(item[i].cpu().numpy(), 'haar')
            LL, (LH, HL, HH) = coeffs2
            temp_LL[i, :, :] = LL
            temp_LH[i, :, :] = LH
            temp_HL[i, :, :] = HL
            temp_HH[i, :, :] = HH

        LL_array.append(temp_LL)
        LH_array.append(temp_LH)
        HL_array.append(temp_HL)
        HH_array.append(temp_HH)

    dis_LL = calculate_distances(LL_array)
    dis_LH = calculate_distances(LH_array)
    dis_HL = calculate_distances(HL_array)
    dis_HH = calculate_distances(HH_array)

    stats['LL'].append(dis_LL)
    stats['LH'].append(dis_LH)
    stats['HL'].append(dis_HL)
    stats['HH'].append(dis_HH)

    plot_distances(dis_LL, dis_LH, dis_HL, dis_HH, path)


# def calculate_distances(array):
#     distances = np.zeros(len(array) - 1) if len(array) > 1 else []
#     for i in range(len(array) - 1):
#         tmp_dis = np.zeros(len(array[i]))
#         for j in range(len(array[i])):
#             tmp_dis[j] = np.linalg.norm(array[i][j] - array[i+1][j])
#         distances[i] = np.mean(tmp_dis)
#     return distances

# def calculate_distances(array):
#     distances = np.zeros(len(array)) if len(array) > 1 else []
#     for i in range(len(array)):
#         tmp_dis = np.zeros(len(array[i]))
#         for j in range(len(array[i])):
#             tmp_dis[j] = np.linalg.norm(array[i][j])
#         distances[i] = np.mean(tmp_dis)
#         distances[i] = np.linalg.norm(array[i])
#     return distances

def calculate_distances(array):
    distances = np.zeros(len(array) - 1) if len(array) > 1 else []
    for i in range(len(array) - 1):
        # dis = np.linalg.norm(array[i] - array[i + 1])
        # dis = np.sum(np.abs(array[i] - array[i + 1]))
        # dis = np.power(np.sum(np.power(np.abs(array[i] - array[i + 1]), 3)), 1/3)
        dis = np.power(np.sum(np.power(np.abs(array[i] - array[i + 1]), 4)), 1/4)
        distances[i] = dis
        # distances[i] = np.linalg.norm(array[i] - array[i + 1])
        # distances[i] = math.sqrt(np.linalg.norm(array[i] - array[i + 1]))
        # distances[i] = math.sqrt(math.sqrt(np.linalg.norm(array[i] - array[i + 1])))
        # distances[i] = np.log(dis / 4096 + 1e-5)
        # distances[i] = np.log10(np.linalg.norm(array[i] - array[i + 1]))
        # distances[i] = np.power(dis, 1e-5)
        # distances[i] = math.sqrt(math.sqrt(math.sqrt(dis)))
        # distances[i] = np.log(dis)
        # distances[i] = math.exp(dis)
        
    return distances

def plot_distances(dis_LL, dis_LH, dis_HL, dis_HH, path):
    # 创建一个图形和两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))  # 两个子图横向排列

    # 第一个子图仅绘制 LL
    x_axis = range(1, len(dis_LL) + 1)
    ax1.plot(x_axis, dis_LL, label='LL Distances', marker='o', color='b')
    ax1.set_title('LL Distances Between Adjacent Elements')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Distance')
    ax1.legend()
    ax1.grid(True)

    # 第二个子图绘制 LH, HL, HH
    x_axis = range(1, len(dis_LH) + 1)  # 确保每个序列都能在图上正确表示
    ax2.plot(x_axis, dis_LH, label='LH Distances', marker='o', color='r')
    ax2.plot(x_axis, dis_HL, label='HL Distances', marker='o', color='g')
    ax2.plot(x_axis, dis_HH, label='HH Distances', marker='o', color='m')
    ax2.set_title('LH, HL, HH Distances Between Adjacent Elements')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Distance')
    ax2.legend()
    ax2.grid(True)

    # 调整子图间距
    plt.tight_layout(pad=3.0)

    # 保存图形到文件
    save_path = path.replace('xt', 'disnorm').replace('.pt', '.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()  # 关闭图形以释放内存

def plot_stats(stats):
    components = ['LL', 'LH', 'HL', 'HH']
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))  # 创建 2x2 的子图网格

    for idx, comp in enumerate(components):
        distances = np.array(stats[comp])
        
        # 如果 distances 是空的或者其中一个维度长度为0，继续下一个循环
        if distances.size == 0 or not distances.shape[1]:
            print(f"{comp} 分量没有可用的数据。")
            continue
        
        max_curve = np.max(distances, axis=0)
        min_curve = np.min(distances, axis=0)
        mean_curve = np.mean(distances, axis=0)
        
        x_axis = range(1, len(mean_curve) + 1)

        ax = axes[idx // 2, idx % 2]  # 获取对应的子图

        ax.plot(x_axis, max_curve, label=f'{comp} Max', marker='o', color='red')
        ax.plot(x_axis, min_curve, label=f'{comp} Min', marker='o', color='green')
        ax.plot(x_axis, mean_curve, label=f'{comp} Mean', marker='o', color='blue')
        ax.set_title(f'{comp} Distance Statistics')
        ax.set_xlabel('Index')
        ax.set_ylabel('Distance')
        ax.legend()
        ax.grid(True)

    plt.tight_layout(pad=3.0)
    plt.savefig('disnorm_batch_stats.png')
    plt.close()

# 初始化统计数据
stats = {'LL': [], 'LH': [], 'HL': [], 'HH': []}

directory = '/data1/fanghaipeng/project/sora/tomesd/SD3_attend/xt'
file_count = 0  # 初始化文件计数器
for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.pt'):
        if file_count < 100:  # 检查是否已达到100个文件的限制
            path = os.path.join(directory, filename)
            process_file(path, stats)
            file_count += 1  # 处理完一个文件后，计数器加1
        else:
            break  # 若已处理100个文件，退出循环

# 绘制统计图
plot_stats(stats)