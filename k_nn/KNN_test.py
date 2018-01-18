# coding=utf-8
import k_nn
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

# 读取训练集样本数据并转换成矩阵
datingDataMat, datingLabels = k_nn.file2matrix("datingTestSet2.txt")
# 将数据归一化
normDataSet, ranges, minVals = k_nn.autoNorm(datingDataMat)
# 实用数据
inArr = array([10, 10000, 0.5])
# 归一化
inArr = (inArr - minVals) / ranges
# 进行分类
print(k_nn.classify0(inArr, normDataSet, datingLabels, 3))

# 测试识别手写数字
k_nn.handwritingClassTest(3)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(normDataSet[:, 0],
#           normDataSet[:, 1],
#           15.0*array(datingLabels),
#           15.0*array(datingLabels))
# plt.show()
