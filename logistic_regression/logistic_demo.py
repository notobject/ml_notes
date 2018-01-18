# encoding=utf-8
# Created by Mr.Long on 2018/1/17 0017.
# 这是文件的概括

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def createDataSet():
    iris = load_iris()

    datasets = []
    labels = []

    for i, data in enumerate(iris.data):
        if iris.target[i] == 0:
            continue
        datasets.append([1.0, iris.data[i, 0], iris.data[i, 2]])
        if iris.target[i] == 1:
            labels.append(0)
        else:
            labels.append(1)
    return datasets, labels


# 梯度下降算法
def gradDscent(datasets, labels, step, lr=0.001):
    dataMat = np.mat(datasets)
    labelMat = np.mat(labels).transpose()

    # m 样本数, n 特征数
    m, n = dataMat.shape
    # 回归系数初始化
    weights = np.ones((n, 1))
    lossArr = []
    weightsArr = []
    # 梯度下降 优化
    for s in range(step):
        y_ = sigmoid(dataMat * weights)
        loss = -(labelMat - y_)
        loss_value = np.mean(loss)
        lossArr.append(loss_value)
        weightsArr.append(weights)
        if s % 100 == 0:
            print("with step {} the loss_value is: {}".format(s, loss_value))
        weights -= lr * dataMat.transpose() * loss

    return weights, lossArr


# 随机梯度下降
def RandGradDscent(datasets, labels, step, lr=0.01):
    datasets = np.array(datasets)
    m, n = np.shape(datasets)

    weights = np.ones(n, dtype=np.float64)
    for j in range(step):
        dataIndex = np.arange(m)
        for i in range(m):
            # 动态调整学习速率
            alpha = 4 / (i + j + 1.0) + lr
            # 随机选择训练样本
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            y_ = sigmoid(np.sum(datasets[randIndex] * weights))
            loss_value = -(labels[randIndex] - y_)
            weights -= loss_value * alpha * datasets[randIndex]
            np.delete(dataIndex, randIndex)
    return weights


def main():
    datasets, labels = createDataSet()
    weights = RandGradDscent(datasets, labels, step=100, lr=0.01)
    weights2,_ = gradDscent(datasets, labels, step=100, lr=0.001)
    print(weights, weights2)

    colors = "gr"
    plt.figure()
    plt.subplot(211)
    plt.title("RandGradDscent")
    for i, data in enumerate(datasets):
        plt.plot(data[1], data[2], ".%s" % colors[labels[i]])
    x = np.arange(5.0, 8.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    plt.plot(x, y, "-")

    plt.subplot(212)
    plt.title("GradDscent")
    for i, data in enumerate(datasets):
        plt.plot(data[1], data[2], ".%s" % colors[labels[i]])
    x = np.arange(5.0, 8.0, 0.1)
    y = (-weights2[0] - weights2[1] * x) / weights2[2]
    plt.plot(x, y, "-")
    # plt.plot(np.arange(0, len(lossArr)), lossArr, ".")

    plt.show()
    pass


if __name__ == "__main__":
    main()
