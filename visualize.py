
"""
给定文件夹，将里面所有npy文件绘制成直方图
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import os
# 设置matplotlib正常显示中文和负号
# matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号


dir_path = 'w2a2'

files= os.listdir(dir_path) #得到文件夹下的所有文件名称
for file in files:
    if '.jpg' in file:
        os.remove(dir_path + '/' + file)
for file in files:
    if '.npy' in file:
        b = np.load(dir_path + '/' + file) 
        b = b.reshape(-1) 
        """
        绘制直方图
        data:必选参数，绘图数据
        bins:直方图的长条形数目，可选项，默认为10
        density:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
        facecolor:长条形的颜色
        edgecolor:长条形边框的颜色
        alpha:透明度
        """
        plt.figure()
        plt.hist(b, bins=20, density=1, facecolor="blue", edgecolor="black", alpha=1)
        # 显示横轴标签
        plt.xlabel("data")
        # 显示纵轴标签
        plt.ylabel("Frequency")
        # 显示图标题
        plt.title(file)
        plt.savefig(dir_path + '/' + os.path.splitext(file)[0] + '.jpg')   #图片的存储
        plt.close()   #关闭matplotlib



