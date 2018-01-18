# coding=utf-8
from numpy import *
import operator
from os import listdir


# K-近邻算法 实现

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    # 计算欧式距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    # 选择距离最小的k个点
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createDataSet():
    # 特征值
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    # 分类
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(filename):
    # 从文件中读取数据 转换为矩阵
    fr = open(filename)
    arrarLines = fr.readlines()
    #
    numberOfLines = len(arrarLines)
    #
    returnMat = zeros((numberOfLines, 3))
    classLabelVextor = []
    index = 0
    # 循环处理每行数据
    for line in arrarLines:
        # 截取掉所有的回车符
        line = line.strip()
        # 分割数据得到 元列表
        listFromLine = line.split('\t')
        # 将得到的元素列表的前三个放入特征矩阵
        returnMat[index, :] = listFromLine[0:3]
        # 将最后一个元素放入标签向量中
        classLabelVextor.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVextor


def autoNorm(dataSet):
    # 归一化特征值 (将特征矩阵中的值都取为0-1区间)
    # 公式 newValue = (oldValue-min)/(max-min)

    # 得到每一列的最小值
    minVals = dataSet.min(0)
    # 得到每一列的最大值
    maxVals = dataSet.max(0)
    # 得到每一列的值域
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    # 当前值 - 最小值
    normDataSet = dataSet - tile(minVals, (m, 1))
    # 再除以值域
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def img2vector(filename):
    # 从文件中读取数据转换成图像向量
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest(k):
    #
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        cr = classify0(vectorUnderTest, trainingMat, hwLabels, k)
        # print "cr = %d,the real=%d" % (cr, classNumStr)
        if (cr != classNumStr):
            errorCount += 1.0
    print("[%d]测试结果:错误= %f ,错误率= %f" % (k, errorCount, errorCount / float(mTest)))
