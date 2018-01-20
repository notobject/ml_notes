# encoding=utf-8
# Created by Mr.Long on 2018/1/20 0020.
# 岭回归
import numpy as np
import math
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


def ridgeRegession(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    denom = xTx + np.eye(xMat.shape[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("denom cannot do inverse.")
        return
    ws = denom.I * (xMat.T * yMat)
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, xMat.shape[1]))
    for i in range(numTestPts):
        ws = ridgeRegession(xMat, yMat, math.exp(i - 10))
        wMat[i, :] = ws.T
    return wMat


def main():
    abX, abY = loadDataSet("abalone.txt")
    ridgeWeghts = ridgeTest(abX, abY)

    plt.figure()
    plt.plot(ridgeWeghts)
    plt.show()
    pass


if __name__ == "__main__":
    main()
