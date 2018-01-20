# encoding=utf-8
# Created by Mr.Long on 2018/1/20 0020.
# 局部加权线性回归( locally weighted linear regression)
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


def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = xMat.shape[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        # 计算测试点与样本的距离
        diffMat = testPoint - xMat[j, :]
        # 利用高斯核函数 更新权重矩阵
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k ** 2))

    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("xTx cannot do inverse.")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat


def main():
    xArr, yArr = loadDataSet("ex0.txt")
    yHat = lwlrTest(xArr, xArr, yArr, 0.01)
    xMat = np.mat(xArr)
    sortedIndex = xMat[:, 1].argsort(0)
    xSorted = xMat[sortedIndex][:, 0, :]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSorted[:, 1], yHat[sortedIndex])

    ax.scatter(xMat[:, 1].flatten().A[0], np.mat(yArr).T.flatten().A[0], s=2, c="red")

    plt.show()
    pass


if __name__ == "__main__":
    main()
