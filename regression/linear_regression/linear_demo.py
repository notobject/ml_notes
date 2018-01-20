# encoding=utf-8
# Created by Mr.Long on 2018/1/20 0020.
# 标准线性回归
import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    numFeat = len(open(filename).readline().split("\t")) - 1
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split("\t")
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegression(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    # 如果xTx的行列式为0 则xTx为奇异矩阵,逆不存在
    if np.linalg.det(xTx) == 0.:
        print("xTx.I cannot do inverse")
        return
    # w的最优估计->详见公式
    ws = xTx.I * (xMat.T * yMat)
    return ws


def main():
    xArr, yArr = loadDataSet("ex0.txt")
    print(xArr[0:2])

    ws = standRegression(xArr, yArr)
    print(ws)

    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    yPredict = xMat * ws

    # 计算预测值与实际值的相关系数
    val = np.corrcoef(yPredict.T, yMat)
    print(val)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    xCopy = xMat.copy()
    xCopy.sort(0)
    yPredict = xCopy * ws
    ax.plot(xCopy[:, 1], yPredict)
    plt.show()
    pass


if __name__ == "__main__":
    main()
