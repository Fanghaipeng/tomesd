# import torch
# import numpy as np
# import pywt
# import matplotlib.pyplot as plt
# import os
# import math
# from tqdm import tqdm


# def process_and_plot(filepath):
#     """
#     加载 .pt 文件，将数据转换为 50x24 的数组，并绘制三维散点图，并将其保存到指定路径。

#     参数:
#         filepath (str): 要加载的 .pt 文件的路径。
#         save_path (str): 图形保存的文件路径。
#     """
#     try:
#         # 加载 .pt 文件
#         data = torch.load(filepath)
#         save_path = filepath.replace('.pt', '.png').replace('xt_avg', 'xt_avg_png')
#         # 转换数据类型
#         if isinstance(data, list):
#             # 确保数据长度为 1200
#             if len(data) == 1200:
#                 # 转换为 numpy 数组，并重塑为 50x24 形状
#                 data_array = np.array(data).reshape(50, 24)
                
#                 # 生成三维散点图所需的 x, y, z 坐标
#                 x = np.arange(50)
#                 x = x[::-1]  # 反转 x 轴    
#                 y = np.arange(24)
#                 X, Y = np.meshgrid(x, y)
                
#                 # 创建图形和轴
#                 fig = plt.figure(figsize=(12, 6))
#                 ax = fig.add_subplot(111, projection='3d')
                
#                 # 绘制散点图
#                 scatter = ax.scatter(X, Y, data_array.T, c=data_array.T, cmap='viridis')
                
#                 # 设置轴标签
#                 ax.set_xlabel('X')
#                 ax.set_ylabel('Y')
#                 ax.set_zlabel('Z')

#                 ax.set_xlim(50, 0)  # 设置 x 轴范围从 50 到 0

#                 # 设置坐标轴的比例 - 使得 x, y, z 轴的单位长度相同
#                 ax.set_box_aspect([2, 1, 1])  # [x 轴长度, y 轴长度, z 轴长度]

#                 # 添加颜色条
#                 fig.colorbar(scatter, ax=ax, label='Data Values')
                
#                 # 保存图形
#                 plt.savefig(save_path)
#                 plt.close(fig)
                
#                 # print(f"图形已保存到: {save_path}")
#             else:
#                 print("列表长度不为1200，无法重塑为50x24形状")
#         else:
#             print("数据不是列表类型")

#     except Exception as e:
#         print(f"处理文件时出错: {e}")


# directory = '/data1/fanghaipeng/project/sora/tomesd/SD3_attend/xt_avg'
# os.makedirs(directory.replace('xt_avg', 'xt_avg_png'), exist_ok=True)
# file_count = 0  # 初始化文件计数器
# for filename in tqdm(os.listdir(directory)):
#     if filename.endswith('.pt'):
#         if file_count < 100:  # 检查是否已达到100个文件的限制
#             path = os.path.join(directory, filename)
#             process_and_plot(path)
#             file_count += 1  # 处理完一个文件后，计数器加1
#         else:
#             break  # 若已处理100个文件，退出循环



import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import os

def process_and_plot_single(filepath, save_path):
    """
    加载.pt文件，将数据转换为50x24的数组，并绘制三维散点图并保存到指定路径。
    """
    try:
        data = torch.load(filepath)
        if isinstance(data, list) and len(data) == 1200:
            data_array = np.array(data).reshape(50, 24)
            plot_data(data_array, save_path)
        else:
            print("数据不是列表类型或长度不为1200")
    except Exception as e:
        print(f"处理文件时出错: {e}")

def plot_data(data_array, save_path):
    """
    使用给定的数据绘制三维散点图并保存。
    """
    x = np.arange(50)
    x = x[::-1]  # 反转 x 轴
    y = np.arange(24)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X, Y, data_array.T, c=data_array.T, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(50, 0)
    ax.set_box_aspect([2, 1, 1])  # 调整视图比例
    fig.colorbar(scatter, ax=ax, label='Data Values')
    plt.savefig(save_path)
    plt.close(fig)

# 设置文件目录
directory = '/data1/fanghaipeng/project/sora/tomesd/SD3_attend/xt_avg'
save_directory = directory.replace('xt_avg', 'xt_avg_png')
os.makedirs(save_directory, exist_ok=True)

# 收集所有数据以计算平均值
all_data = []
file_count = 0  # 初始化文件计数器
for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.pt'):
        if file_count < 100:  # 限制处理文件数量
            path = os.path.join(directory, filename)
            save_path = path.replace('.pt', '.png').replace('xt_avg', 'xt_avg_png')
            process_and_plot_single(path, save_path)
            data_array = torch.load(path)
            if isinstance(data_array, list) and len(data_array) == 1200:
                all_data.append(np.array(data_array).reshape(50, 24))
            file_count += 1  # 处理完一个文件后，计数器加1
        else:
            break

# 计算所有数据的平均值并绘制总图
if all_data:
    average_data = np.mean(all_data, axis=0)
    plot_data(average_data, '/data1/fanghaipeng/project/sora/tomesd/SD3_attend/average_avg.png')