
"""
给定文件夹，将里面所有npy文件绘制成直方图
"""

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
import os
# 设置matplotlib正常显示中文和负号
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号


dir_path = 'w4a2'
files= os.listdir(dir_path) #得到文件夹下的所有文件名称
for file in files:
    if '.npy' in file:
        b = np.load(dir_path + '/' + file) 
        b = b.reshape(-1) 
        """
        绘制直方图
        data:必选参数，绘图数据
        bins:直方图的长条形数目，可选项，默认为10
        normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
        facecolor:长条形的颜色
        edgecolor:长条形边框的颜色
        alpha:透明度
        """
        plt.hist(b, bins=50, density=1, facecolor="blue", edgecolor="black", alpha=0.7)
        # 显示横轴标签
        plt.xlabel("区间")
        # 显示纵轴标签
        plt.ylabel("频数/频率")
        # 显示图标题
        plt.title("频数/频率分布直方图")
        print(file)
        plt.savefig(dir_path + '/' + os.path.splitext(file)[0] + '.jpg')   #图片的存储
plt.close()   #关闭matplotlib


