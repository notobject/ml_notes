# encoding=utf-8
# Created by Mr.Long on 2018/1/20 0020.
# 回归树/模型树

import numpy as np


def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split("\t")
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


# 二元法切分数据集
def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


# 解线性方程
def linearSolve(dataSet):
    m, n = np.shape(dataSet)
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    X[:, 1:n] = dataSet[:, 0:n - 1]
    Y = dataSet[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0:
        print("cannot do inverse. try increasing sencond value of ops.")
        return
    ws = xTx.I * (X.T * Y)
    return ws, X, Y


# 返回线性回归模型的回归系数作为叶子节点
def modelLeaf(dataSet):
    ws, X, Y = linearSolve(dataSet)
    return ws


# 计算线性回归模型的拟合误差
def modelErr(dataSet):
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    return np.sum(np.power(Y - yHat, 2))


# 生成叶子节点函数-回归
def regLeaf(dataSet):
    return np.mean(dataSet[:, -1].T.A.tolist()[0])


# 计算总方差函数-回归
def regErr(dataSet):
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


# 选择最优划分特征,和特征的取值
def chooseBestSplit(dataSet, leafType=regLeaf, errTpye=regErr, ops=(1, 4)):
    tolS = ops[0]
    tolN = ops[1]
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)
    S = errTpye(dataSet)
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        # print(dataSet[:, featIndex].T)
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
                continue
            newS = errTpye(mat0) + errTpye(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if S - bestS < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 树构建
def createTree(dataSet, leafType=regLeaf, errTpye=regErr, ops=(1, 4)):
    feat, val = chooseBestSplit(dataSet, leafType, errTpye, ops)
    if feat == None:
        return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errTpye, ops)
    retTree['right'] = createTree(rSet, leafType, errTpye, ops)
    return retTree


# 判断对象是否为树
def isTree(obj):
    return type(obj).__name__ == 'dict'


# 从从至下递归寻找两个叶节点,计算他们的均值
def getMean(tree):
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    return (tree['left'] + tree['right']) / 2.0


# 根据树在测试数据上的误差 进行后剪枝处理
def prune(tree, testData):
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], lSet)
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['left'], 2)) + \
                       np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge < errorNoMerge:
            return treeMean
        else:
            return tree
    else:
        return tree


def main():
    print("--------------------------回归树-----------------------")
    dataSet = loadDataSet("ex2.txt")
    dataMat = np.mat(dataSet)
    tree = createTree(dataMat, ops=(0, 1))
    print(tree)

    testDataSet = loadDataSet("ex2test.txt")
    testDataMat = np.mat(testDataSet)
    prune(tree, testDataMat)
    print(tree)

    print("--------------------------模型树-----------------------")
    dataSet = loadDataSet("exp2.txt")
    dataMat = np.mat(dataSet)
    tree = createTree(dataMat, modelLeaf, modelErr, ops=(1, 4))
    print(tree)
    pass


if __name__ == "__main__":
    main()
