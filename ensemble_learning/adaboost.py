# encoding=utf-8
# Created by Mr.Long on 2018/1/19 0019.
# 基于单层决策树的AdaBoost算法
import numpy as np


# 简单测试数据集
def loadSimpData():
    dataMat = np.matrix([[1., 2.1],
                         [2., 1.1],
                         [1., 1.],
                         [1., 1.],
                         [2., 1.], ])
    classLabes = [1., 1., -1., -1., 1.]
    return dataMat, classLabes


# 单层决策树分类器
def singleLayerDTClassify(dataMat, dim, threshVal, threshIneq):
    # 先将所有的类别设为1
    resArr = np.ones((np.shape(dataMat)[0], 1))
    # 对于给定的特征(dim)的值与给定阈值(threshVal)比较 大小(threshIneq)
    if threshIneq == 'lt':
        # 如果给定特征的值小于等于阈值,则分类为 -1 类别
        resArr[dataMat[:, dim] <= threshVal] = -1
    else:
        # 如果给定特征的值大于阈值,则分类为 -1 类别
        resArr[dataMat[:, dim] > threshVal] = -1
    return resArr


# 基于特征权重向量D 训练单层决策树,得到最佳分类器
def singeLayerDTtrain(dataArr, classLabels, D):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(classLabels).T
    m, n = dataMat.shape
    tranSteps = 10.
    bestSingleLayerDT = {}
    bestClassifyRes = np.mat(np.zeros((m, 1)))
    minError = np.inf

    # 对每一个特征
    for i in range(n):
        # 计算阈值增长的步长
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / tranSteps
        for j in range(-1, int(tranSteps) + 1):
            for ineqal in ['lt', 'gt']:
                # 阈值每次增长stepSize
                threshVal = (rangeMin + float(j)) * stepSize
                # 将第i个特征 根据不等号inequal 与threshVal进行比较后 得到分类结果
                predictRes = singleLayerDTClassify(dataMat, i, threshVal, ineqal)
                # 计算分类结果的加权错误率
                errArr = np.ones((m, 1))
                errArr[predictRes == labelMat] = 0
                weightedError = D.T * errArr
                # print("dim:{}, thresh:{}, ineqal:{}, weightedErro:{}".format(i, threshVal, ineqal, weightedError))
                if weightedError < minError:
                    # 当前的分类器为最优
                    minError = weightedError
                    bestClassifyRes = predictRes.copy()
                    bestSingleLayerDT['dim'] = i
                    bestSingleLayerDT['thresh'] = threshVal
                    bestSingleLayerDT['ineq'] = ineqal
    # 返回最优分类器的参数, 加权错误率, 分类结果
    return bestSingleLayerDT, minError, bestClassifyRes


# 训练AdaBoost
def adaBoostTrain(dataArr, classLabels, numIter=40):
    weakClassifyArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIter):
        print(D)
        bestSingleLayerDT, error, classifyRes = singeLayerDTtrain(dataArr, classLabels, D)

        # 更新分类器权重 alpha
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestSingleLayerDT['alpha'] = alpha
        # 将该分类器加入到弱分类器组中
        weakClassifyArr.append(bestSingleLayerDT)
        # 更新特征权重 D
        expon = np.multiply(-1 * alpha * np.mat(classLabels).T, classifyRes)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        # 分类结果加权求和
        aggClassEst += alpha * classifyRes
        # 计算加权后分类结果的错误率
        aggErros = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErros.sum() / m
        # 如果错误率为0 则结束训练
        if errorRate == 0.0:
            break
    return weakClassifyArr

# adaBoost分类器
def adaBoostClassify(dataToClass, weakClassifys):
    dataMat = np.mat(dataToClass)
    m = dataMat.shape[0]
    aggClassEst = np.mat(np.zeros((m, 1)))

    for i in range(len(weakClassifys)):
        # 用最优弱分类器来分类
        classRes = singleLayerDTClassify(dataMat, weakClassifys[i]["dim"], weakClassifys[i]["thresh"],
                                         weakClassifys[i]["ineq"])
        # 对每一个最优弱分类器的分类结果进行加权求和
        aggClassEst += weakClassifys[i]["alpha"] * classRes
        # print(aggClassEst)
    # 返回加权求和后的最终分类结果
    return np.sign(aggClassEst)


def main():
    dataArr, classLabels = loadSimpData()
    # 用数据集训练一组弱分类器
    weakClassifys = adaBoostTrain(dataArr, classLabels,40)
    print(len(weakClassifys),weakClassifys)

    # 用训练后的弱分类器对测试数据分类
    testRes = adaBoostClassify(dataArr, weakClassifys)
    print(testRes)

if __name__ == "__main__":
    main()
