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
#         save_path = filepath.replace('.pt', '.png').replace('xt_dis', 'xt_dis_png')
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


# directory = '/data1/fanghaipeng/project/sora/tomesd/SD3_attend/xt_dis'
# os.makedirs(directory.replace('xt_dis', 'xt_dis_png'), exist_ok=True)
# file_count = 0  # 初始化文件计数器
# for filename in tqdm(os.listdir(directory)):
#     if filename.endswith('.pt'):
#         if file_count < 10:  # 检查是否已达到100个文件的限制
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

def plot_data(data_array, save_path, plot_3d=False, plot_2d=False):
    """
    使用给定的数据绘制三维散点图并保存。如果启用，还会绘制特定方向的三维视图和二维视图。
    """
    plt.style.use('default')  # 重置样式为默认，确保无其他样式影响
    plt.rcParams['figure.facecolor'] = 'white'  # 设置图表背景色为白色
    plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景色为白色
    plt.rcParams['font.size'] = 24  # 设置全局字体大小


    for i in range(len(data_array)):
        for j in range(len(data_array[i])):
            if data_array[i][j] > 2:
                data_array[i][j] = 2 + (data_array[i][j] - 2) * 0.5
            
    x = np.arange(50)
    x = x[::-1]  # 反转 x 轴
    y = np.arange(24)
    X, Y = np.meshgrid(x, y)
    point_size = 160
    # 创建三维视图
    fig = plt.figure(figsize=(24, 12))
    ax3d = fig.add_subplot(111, projection='3d')
    ax3d.set_facecolor('#FFFFFF')  
    # scatter = ax3d.scatter(X, Y, data_array.T, c=data_array.T, cmap='viridis', s=point_size)
    scatter = ax3d.scatter(X, Y, data_array.T, c=data_array.T, cmap='coolwarm', s=point_size)

    # ax3d.set_xlabel('X')
    # ax3d.set_ylabel('Y')
    # ax3d.set_zlabel('Z')
    ax3d.set_xlim(50, 0)
    ax3d.set_box_aspect([2, 1, 1])  # 调整视图比例
    # fig.colorbar(scatter, ax=ax3d, label='Data Values')
    


    # 设置网格线颜色
    ax3d.xaxis.pane.set_edgecolor('#e0e0e0')
    ax3d.yaxis.pane.set_edgecolor('#e0e0e0')
    ax3d.zaxis.pane.set_edgecolor('#e0e0e0')
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False

    plt.savefig(save_path)
    plt.close(fig)
    save_path = save_path.replace('3d.png', '2d.png')
    if plot_2d:
        plt.rcParams['font.size'] = 28  # 设置全局字体大小
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111)  # 添加二维 subplot
        # 计算 Y 方向的均值
        Z_mean = np.mean(data_array, axis=1)
        if Z_mean[-1] < 1:
            Z_mean[-1] = Z_mean[-2] + 0.001

        scatter2 = ax.scatter(x, Z_mean, c=Z_mean, cmap='coolwarm', zorder=3, s=point_size)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Average Z Value')
        ax.set_xlim(50, 0)
        ax.set_box_aspect(2/3)
        # fig.colorbar(scatter2, ax=ax, label='Average Z Values')
        
        # 设置轴的网格线颜色，并调整层级
        ax.grid(True)  # 启用网格线
        ax.xaxis.grid(True, color='#e0e0e0', zorder=0)  # 设置 X 轴网格线颜色，并调整层级
        ax.yaxis.grid(True, color='#e0e0e0', zorder=0)  # 设置 Y 轴网格线颜色，并调整层级

        # 设置轴的颜色
        ax.spines['right'].set_color('#808080')  # 设置右轴颜色为浅灰
        ax.spines['top'].set_color('#808080')    # 设置顶部轴颜色为浅灰

        plt.savefig(save_path)
        plt.close(fig)




# def plot_data(data_array, save_path, plot_3d=False, plot_2d=False):
#     """
#     使用给定的数据绘制三维散点图并保存。如果启用，还会绘制特定方向的三维视图和二维视图。
#     """
#     plt.style.use('default')  # 重置样式为默认，确保无其他样式影响
#     plt.rcParams['figure.facecolor'] = 'white'  # 设置图表背景色为白色
#     plt.rcParams['axes.facecolor'] = 'white'  # 设置坐标轴背景色为白色
#     plt.rcParams['font.size'] = 8  # 设置全局字体大小

#     x = np.arange(50)
#     x = x[::-1]  # 反转 x 轴
#     y = np.arange(24)
#     X, Y = np.meshgrid(x, y)
    
#     # 创建三维视图
#     fig = plt.figure(figsize=(12, 6))
#     ax3d = fig.add_subplot(111, projection='3d')
#     # 设置较浅的灰色背景
#     ax3d.set_facecolor('#FFFFFF')  
#     scatter = ax3d.scatter(X, Y, data_array.T, c=data_array.T, cmap='viridis')
#     ax3d.set_xlabel('X')
#     ax3d.set_ylabel('Y')
#     ax3d.set_zlabel('Z')
#     ax3d.set_xlim(50, 0)
#     ax3d.set_box_aspect([2, 1, 1])  # 调整视图比例
#     fig.colorbar(scatter, ax=ax3d, label='Data Values')
#     plot_3d = True
#     plot_2d = True
#     # if plot_3d:
#     #     # 从 X 方向视图
#     #     # ax2 = fig.add_subplot(132, projection='3d')
#     #     # scatter2 = ax2.scatter(X, Y, data_array.T, c=data_array.T, cmap='viridis')
#     #     # ax2.view_init(elev=0, azim=-90)
#     #     # ax2.set_xlabel('Y')
#     #     # ax2.set_ylabel('Z')
#     #     # ax2.set_zlabel('X')
#     #     # ax2.set_xlim(50, 0)
#     #     # ax2.set_title('View from X-axis')

#     #     # 从 Y 方向视图
#     #     ax3 = fig.add_subplot(153, projection='3d')
#     #     scatter3 = ax3.scatter(X, Y, data_array.T, c=data_array.T, cmap='viridis')
#     #     ax3.view_init(elev=10, azim=0)
#     #     ax3.set_xlabel('X')
#     #     ax3.set_ylabel('Z')
#     #     ax3.set_zlabel('Y')
#     #     ax3.set_title('View from Y-axis')

#     plt.savefig(save_path)
#     plt.close(fig)
#     save_path = save_path.replace('3d.png', '2d.png')
#     if plot_2d:
#         fig = plt.figure(figsize=(12, 6))
#         ax = fig.add_subplot(111)  # 添加第二个三维 subplot
#         # 计算 Y 方向的均值
#         Z_mean = np.mean(data_array, axis=1)
#         if Z_mean[-1] < 1:
#             Z_mean[-1] = Z_mean[-2] + 0.001
#         # # 绘制第二个三维散点图，使用 Y 方向的均值
#         # scatter2 = ax3d2.scatter(x, np.zeros_like(x), Z_mean, c=Z_mean, cmap='viridis')
#         # ax3d2.view_init(elev=0, azim=-90)
#         # ax3d2.set_xlabel('X')
#         # ax3d2.set_ylabel('Y')
#         # ax3d2.set_zlabel('Z Average')
#         # ax3d2.set_xlim(50, 0)
#         # ax3d2.set_box_aspect([1.5, 1, 1])
#         # # 隐藏 Y 轴的刻度标签
#         # ax3d2.yaxis.set_tick_params(labelleft=False)
#         # 绘制二维散点图
#         scatter2 = ax.scatter(x, Z_mean, c=Z_mean, cmap='viridis')
#         ax.set_xlabel('X')
#         ax.set_ylabel('Average Z Value')
#         ax.set_xlim(50, 0)
#         ax.set_box_aspect(2/3)
#         fig.colorbar(scatter2, ax=ax, label='Average Z Values')
#         # 设置轴的颜色为非常浅的灰色
#         ax.tick_params(axis='x', colors='#f9f9f9')  # 设置 X 轴刻度颜色
#         ax.tick_params(axis='y', colors='#f9f9f9')  # 设置 Y 轴刻度颜色

#         plt.savefig(save_path)
#         plt.close(fig)
#         # fig.colorbar(scatter2, ax=ax3d2, label='Average Data Values')

#         # import ipdb
#         # ipdb.set_trace()
#         # 添加二维视图，计算 X 和 Y 维度的均值
#         # x_mean = np.mean(data_array, axis=0)
#         # y_mean = np.mean(data_array, axis=1)
#         # y_max = np.max(data_array, axis=1)  
#         # y_min = np.min(data_array, axis=1)

#         # 添加颜色条
#         # fig.colorbar(scatter, ax=ax2d, label='Data Values')
#         # y_mean = np.round(y_mean, 5)
#         # y_max = np.round(y_max, 5)
#         # y_min = np.round(y_min, 5)

#         # print(f"X Mean")
#         # for data_mean in y_mean:
#         #     print(data_mean)
#         # ipdb.set_trace()
#         # print(f"X Max")
#         # for data_max in y_max:
#         #     print(data_max)
#         # ipdb.set_trace()
#         # print(f"X Min")
#         # for data_min in y_min:
#         #     print(data_min)

#         # ax_x = fig.add_subplot(133)
#         # ax_x.plot(x_mean)
#         # ax_x.set_title('Mean across Y dimension')
#         # ax_x.set_xlabel('X')
#         # ax_x.set_ylabel('Mean value')
#         # ax_x.set_box_aspect(2/3)

#         # ax_y = fig.add_subplot(154)
#         # ax_y.plot(y_mean)
#         # ax_y.set_title('Mean across X dimension')
#         # ax_y.set_xlabel('Y')
#         # ax_y.set_ylabel('Mean value')
#         # ax_y.set_box_aspect(2/3)

# 设置文件目录
directory = '/data1/fanghaipeng/project/sora/tomesd/SD3_attend/xt_dis'
save_directory = directory.replace('xt_dis', 'xt_dis_png')
os.makedirs(save_directory, exist_ok=True)

# 收集所有数据以计算平均值
all_data = []
file_count = 0  # 初始化文件计数器
for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.pt'):
        if file_count < 10:  # 限制处理文件数量
            path = os.path.join(directory, filename)
            save_path = path.replace('.pt', '.png').replace('xt_dis', 'xt_dis_png')
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
    average_save_path = '/data1/fanghaipeng/project/sora/tomesd/SD3_attend/average_dis_3d.png'
    # plot_data(average_data, average_save_path)
    plot_data(average_data, average_save_path, plot_3d=True, plot_2d=True)  # 现在传递 plot_2d=True

# 设置文件目录
directory = '/data1/fanghaipeng/project/sora/tomesd/SD3_attend/xt_avg'
save_directory = directory.replace('xt_avg', 'xt_avg_png')
os.makedirs(save_directory, exist_ok=True)

# 收集所有数据以计算平均值
all_data = []
file_count = 0  # 初始化文件计数器
for filename in tqdm(os.listdir(directory)):
    if filename.endswith('.pt'):
        if file_count < 10:  # 限制处理文件数量
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
    average_save_path = '/data1/fanghaipeng/project/sora/tomesd/SD3_attend/average_avg_3d.png'
    # plot_data(average_data, average_save_path)
    plot_data(average_data, average_save_path, plot_3d=True, plot_2d=True)  # 现在传递 plot_2d=True