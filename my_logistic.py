#-*- coding=utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def logistic(data):  #    逻辑回归主体函数
    m = data.shape[0]        # m为样本数量
    data.insert(0,'ones',np.ones(m))   # 每一行第一列插入一个1

    #特征缩放
    data.iloc[:,1] = (data.iloc[:,1]-np.mean(data.iloc[:,1]))/(max(data.iloc[:,1])-min(data.iloc[:,1]))
    data.iloc[:,2] = (data.iloc[:,2] - np.mean(data.iloc[:,2])) / (max(data.iloc[:,2]) - min(data.iloc[:, 2]))
    # 初始化theta和损失函数的值
    theta = [0.5,0.5,0.5]
    J = cost(m,theta,data.iloc[:,1:4])
    count = 0  # 记录迭代次数
    while count <10000:
        temp = []  # 为了实现同时更新所有theta，设立一个暂时存放的列表
        for j in range(len(theta)):
            temp.append(update_theta(0.001,m,theta,j,data.iloc[:,0:3],data.iloc[:,3]))   # 计算所有更新的theta值，存入temp
        for i in range(len(theta)):  # 更新所有的theta
            theta[i] = temp[i]
        print(theta)
        J = cost(m,theta,data.iloc[:,1:4])   # 更新完theta后重新计算损失函数值
        count += 1
        print(J,count)

    # 画出拟合的边界线
    x = np.arange(min(data.iloc[:,1]),max(data.iloc[:,1]),0.1)
    y = -(theta[1]/theta[2]*x+theta[0]/theta[2])
    plt.plot(x,y)

    # 画出训练样本点
    x = data.iloc[:,1]
    z = data.iloc[:,2]
    y = data.iloc[:,3]
    for i in range(m):
        if y[i] == 0:
            plt.scatter(x[i],z[i],marker='.',c='k')
        else:
            plt.scatter(x[i],z[i],marker='.',c='c')
    plt.figure(figsize=(12,16))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.show()

def sigmoid(z):   # 核函数
    return 1 / (1 + np.exp(-z))

def cost(m,theta,data):  # 计算损失函数值，输入参数有：样本数，theta所有值的列表，去掉第一列（全为1）之后的数据
    x = data.iloc[:,0]   # x1特征数据
    y = data.iloc[:,2]   # 标签数据
    z = data.iloc[:,1]   # x2特征数据
    result = []
    for i in range(m):   # 分别计算所有累加项的值，存入列表
        a = theta[2]*z[i]+theta[1]*x[i]+theta[0]
        h = sigmoid(a)
        if y[i] == 1:
            result.append(-np.log(h))
        else:
            result.append(-np.log(1-h))
    result = sum(result)
    return -result/m+1/(2*m)*((np.array(theta)**2).sum())

def update_theta(a,m,theta,j,x,y):  # 更新theta值，输入参数：步长，样本数，theta所有值的列表，需要更新的theta下标，所有特征项的数据，所有标签项的数据
    result = []
    for i in range(m):   # 分别计算所有累加项的值，存入列表
        a = theta[2] * x.iloc[i, 2] + theta[1] * x.iloc[i, 1] + theta[0]
        h = sigmoid(a)
        result.append((h-y[i])*x.iloc[i,j])
    result = sum(result)
        if j != 0:
        return theta[j]-a/m*(result+theta[j])
    else:
        return theta[j]-a/m*result

data = pd.read_csv('ex2data1.txt')



logistic(data)
