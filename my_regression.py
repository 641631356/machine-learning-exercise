# -*-coding=utf-8-*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def line_regression(data): # 线性回归主体函数
    m = data.shape[0]   # m为样本数量
    data.insert(0,'ones',np.ones(m)) # 插入全为1的一列
    # 初始化theta值和损失函数值
    theta = [1,1]
    J = cost(m,theta,data.iloc[:,1:3])
    count = 0    # 统计迭代次数
    while count <10000:
        temp = []    # 实现同步更新的暂存列表
        for j in range(len(theta)):    # 计算所有theta更新值存入列表
            temp.append(update_theta(0.001,m,theta,j,data.iloc[:,0:2],data.iloc[:,2]))
        for i in range(len(theta)):   # 更新所有theta
            theta[i] = temp[i]
        print(theta)
        J = cost(m,theta,data.iloc[:,1:3])  # 更新theta后重新计算损失函数值
        count += 1
        print(J,count)
    # 画出边界线
    x = np.arange(min(data.iloc[:,1]),max(data.iloc[:,1]),0.1)
    y = theta[1]*x+theta[0]
    plt.plot(x,y)

    # 画出所有样本点
    x = data.iloc[:,1]
    y = data.iloc[:,2]
    plt.scatter(x,y,marker='.')
    plt.figure(figsize=(12,16))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()


def cost(m,theta,data):  # 计算损失函数，输入参数：样本数，theta所有值的列表，除去第一列（全为1）的数据
    x = data.iloc[:,0]   # 特征数据
    y = data.iloc[:,1]   # 标签数据
    result = []
    for i in range(m): # 分别计算每一个累加项，存入列表
        result.append(((theta[1]*x[i]+theta[0])-y[i])**2)
    minors = sum(result)
    return minors/(2*m)+1/(2*m)*((np.array(theta)**2).sum())

def update_theta(a,m,theta,j,x,y):   # 计算更新theta值，输入参数：步长，样本数，theta所有值的列表，要更新的theta下标，所有特征数据，所有标签数据
    minors = []
    for i in range(m):    # 分别计算所有累加项的值，存入列表
        minors.append(((theta[1]*x.iloc[i][1]+theta[0])-y[i])*x.iloc[i][j])
    minors = sum(minors)
    return theta[j]-a/m*(minors+theta[j])

data = pd.read_csv('ex1data1.txt',names=['population','profit'])



line_regression(data)