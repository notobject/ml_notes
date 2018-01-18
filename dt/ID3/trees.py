#coding=utf-8
from numpy import *
from math import log
import treePlotter
import sys
import os

#解决中文乱码问题
# reload(sys)
# sys.setdefaultencoding('utf-8')

def calcShannonEnt(dataSet):

    #计算数据集的香农嫡

    #得到数据集中样本总数
    numEntries = len(dataSet)
    #创建数据字典，键值为数据集中最后一列的值
    labelCounts = {}
    #循环数据集的每一行（即每一个样本实例）
    for featVec in dataSet:
        #每一行的最后一列 是 该样本的分类标签
        currentLabel = featVec[-1]

        #如果字典中不存在该分类，则创建个字典，出现的次数+1
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
            labelCounts[currentLabel] += 1
        shannonEnt = 0.0
    #循环每个特征属性，求香农嫡
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    # 1.将数据集中所有axis位置的属性不为value的数据丢弃，
    # 2.如果是 去除该特征
    # 返回新的数据集

    # dataSet 待划分的数据集
    # axis    划分数据集的特征
    # value   需要返回的特征的值
    retDataSet = []

    #循环数据集的每一行
    for featVec in dataSet:
        #如果该数据符合要求的特征
        if featVec[axis] == value:
            #抽取除该特征以外的所有元素 放入新创建的列表

            # featVec[:axis]返回 0 - axis-1 的元素列表
            reducedFeatVec = featVec[:axis]
            # featVec[axis + 1:] 返回 axis 以后所有的元素列表
            reducedFeatVec.extend(featVec[axis + 1:])
            # 以上两行 将除axis位置以外的元素放进了 reducedFeatVec

            # 放入新创建的列表
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    #
    # 特征数量（ 最后一列是分类标签，所以减一）
    numFeatures = len(dataSet[0]) - 1
    # 原始数据集的香农嫡
    baseEntropy = calcShannonEnt(dataSet)
    # 最好信息增益
    bestInfoGain = 0.0
    # 最好划分特征
    bestFeature = -1
    for i in range(numFeatures):
        # 得到数据集中i特征的所有可能取值
        featList = [example[i] for example in dataSet]

        # 利用set的唯一性去重
        uniqueVals = set(featList)

        newEntropy = 0.0
        # 计算划分后的香农嫡
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy -= prob * calcShannonEnt(subDataSet)
            #信息增益 = 原始香农嫡 - 划分后新数据集的香农嫡
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    #返回在数据集中出现频率最高的分类名称
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),
                              reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):

    #得到数据集中的分类列表名称
    classList = [example[-1] for example in dataSet]

    # 类别完全相同则停止继续划分
    # classList.count(classList[0]) 统计列表中第0个类别的数量
    # len(classList) 统计类别总数量
    # 两者相同说明列表中只有一种类别了
    if classList.count(classList[0]) == len(classList):
        return classList[0]

    # 遍历完所有特征后返回出现频率最高的类别名称
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择最优的划分特征
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    #删除已使用的最优特征
    del(labels[bestFeat])
    #得到所有最优划分属性的取值
    featValues = [example[bestFeat] for example in dataSet]
    #去重
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 复制特征标签列表,用于递归
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


# 使用决策树的分类函数
def classify(inputTree, featLabels, testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    classLabel = 'null'
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(inputTree, fw)
    fw.close()


def geabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)


if __name__ == '__main__':
    isExistsTree = os.path.exists("myTree.txt")
    lensesLabels = [u'ax', u'ay', u'az', u'gx',u'gy',u'gz',u'mx',u'my',u'mz']
    if isExistsTree:
        myTree = geabTree("myTree.txt")
    else:
        fr = open('0_0.txt')
        lenses = [inst.strip().split(',') for inst in fr.readlines()]
        myTree = createTree(lenses, lensesLabels)
    treePlotter.createPlot(myTree)
    # storeTree(myTree, "myTree.txt")
    # textVec = [u'x', u'y', u'z', u'A']
    # res = classify(myTree, lensesLabels, textVec)
    # print res
