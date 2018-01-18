# encoding=utf-8
# Created by Mr.Long on 2018/1/17 0017.
# SMO算法的数据存储结构
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pickle


class SMO_SVM:
    """
    """

    """
        dataMatIn:      数据集
        classLables:    标签
        C          :    常数C
        toler      :    容错率
    """

    def __init__(self, dataMatIn, classLables, C, toler, maxIter, kTup=('lin', 0)):
        # 样本数据
        self.X = np.mat(dataMatIn)
        # 样本标签
        self.labelMat = np.mat(classLables).transpose()
        # 常数C 控制软间隔
        self.C = C
        # 容错率
        self.tol = toler
        # 样本数
        self.m = np.shape(dataMatIn)[0]
        # 优化目标
        self.alphas = np.mat(np.zeros((self.m, 1)))
        # 偏移量
        self.b = 0
        # 误差缓存矩阵
        self.eCahce = np.mat(np.zeros((self.m, 2)))
        # 最大迭代次数
        self.maxIter = maxIter
        # 核函数配置
        self.kTup = kTup
        self.sVectorIndex = []
        self.K = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.K[:, i] = self.kernelTrans(self.X, self.X[i, :])

    """
        计算误差
    """

    def calcEk(self, k):
        # 计算预测值
        # fXK = float(np.multiply(self.alphas, self.labelMat).T * (self.X * self.X[k, :].T)) + self.b
        fXK = float(np.multiply(self.alphas, self.labelMat).T * self.K[:, k] + self.b)
        # 计算预测值与真实值的误差
        EK = fXK - float(self.labelMat[k])
        return EK

    """
        选择能使得误差最大化的另一个要优化的alpha
    """

    def selectJ(self, i, Ei):
        maxK = -1
        maxDeltaE = 0
        Ej = 0
        self.eCahce[i] = [1, Ei]
        validEcacheList = np.nonzero(self.eCahce[:, 0].A)[0]
        if len(validEcacheList) > 1:
            # 如果有误差缓存,则从误差缓存中选取误差最大的一个
            for k in validEcacheList:
                if k == i:
                    continue
                Ek = self.calcEk(k)
                deltaE = abs(Ei - Ek)
                if (deltaE > maxDeltaE):
                    maxK = k
                    maxDeltaE = deltaE
                    Ej = Ek
            return maxK, Ej
        else:
            # 否则随机选取一个
            j = self.selectJrand(i, self.m)
            Ej = self.calcEk(j)
            return j, Ej

    """
        更新误差缓存
    """

    def updateEk(self, k):
        # 计算预测样本k的误差
        Ek = self.calcEk(k)
        # 将误差缓存起来
        self.eCahce[k] = [1, Ek]

    """
        随机选择另一个alpha
    """

    def selectJrand(self, i, m):
        j = i
        while j == i:
            j = int(np.random.uniform(0, m))
        return j

    """
        将alpha的值控制在L-H的范围内
    """

    def clipAlpha(self, aj, H, L):
        if aj > H:
            aj = H
        if L > aj:
            aj = L
        return aj

    def innerL(self, i):
        Ei = self.calcEk(i)
        if ((self.labelMat[i] * Ei < -self.tol) and (self.alphas[i] < self.C)) or \
                ((self.labelMat[i] * Ei > self.tol) and (self.alphas[i] > 0)):
            j, Ej = self.selectJ(i, Ei)
            alphaIold = self.alphas[i].copy()
            alphaJold = self.alphas[j].copy()
            if self.labelMat[i] != self.labelMat[j]:
                L = max(0, self.alphas[j] - self.alphas[i])
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                H = min(self.C, self.alphas[j] + self.alphas[i])
            if L == H:
                # print("L == H -> L: {}, H: {}".format(L, H))
                return 0
            # eta = 2.0 * self.X[i, :] * self.X[j, :].T - self.X[i, :] * self.X[i, :].T - self.X[j, :] * self.X[j, :].T
            eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                # print("eta == {} >= 0".format(eta))
                return 0
            self.alphas[j] -= self.labelMat[j] * (Ei - Ej) / eta
            self.alphas[j] = self.clipAlpha(self.alphas[j], H, L)
            self.updateEk(j)
            if abs(self.alphas[j] - alphaJold) <= 0.00001:
                # print("j is not move enough")
                return 0
            self.alphas[i] += self.labelMat[j] * self.labelMat[i] * (alphaJold - self.alphas[j])
            self.updateEk(i)
            # b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.X[i, :] * self.X[i, :].T - \
            #      self.labelMat[j] * (self.alphas[j] - alphaJold) * self.X[i, :] * self.X[j, :].T
            #
            # b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.X[i, :] * self.X[j, :].T - \
            #      self.labelMat[j] * (self.alphas[j] - alphaJold) * self.X[j, :] * self.X[j, :].T
            b1 = self.b - Ei - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.K[i, i] - \
                 self.labelMat[j] * (self.alphas[j] - alphaJold) * self.K[i, j]
            b2 = self.b - Ej - self.labelMat[i] * (self.alphas[i] - alphaIold) * self.K[i, j] - \
                 self.labelMat[j] * (self.alphas[j] - alphaJold) * self.K[j, j]
            if (0 < self.alphas[i]) and (self.C > self.alphas[i]):
                self.b = b1
            elif (0 < self.alphas[j]) and (self.C > self.alphas[j]):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def train(self):
        iter = 0
        entirSet = True
        alphaPairsChanged = 0
        while (iter < self.maxIter) and ((alphaPairsChanged > 0) or entirSet):
            alphaPairsChanged = 0
            if entirSet:
                for i in range(self.m):
                    alphaPairsChanged += self.innerL(i)
                # print("fullSet, iter :%d i:%d, changed:%d" % (iter, i, alphaPairsChanged))
                iter += 1
            else:
                nonBoundIs = np.nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += self.innerL(i)
                # print("non-bound, iter :%d i:%d, changed:%d" % (iter, i, alphaPairsChanged))
                iter += 1
            if entirSet:
                entirSet = False
            elif alphaPairsChanged == 0:
                entirSet = True
                # print("iter:{}".format(iter))

        self.sVectorIndex = np.nonzero(self.alphas > 0)[0]
        # print("训练完成,共有 %d 个支持向量" % len(self.sVectorIndex))
        return self.b, self.alphas, self.sVectorIndex

    def kernelTrans(self, X, A):
        m, n = np.shape(X)
        K = np.mat(np.zeros((m, 1)))
        if self.kTup[0] == 'lin':
            K = X * A.T
        elif self.kTup[0] == 'rbf':
            for j in range(m):
                deltaRow = X[j, :] - A
                K[j] = deltaRow * deltaRow.T
            K = np.exp(K / (-1 * self.kTup[1] ** 2))
        else:
            raise NameError('不支持的核函数类型')
        return K

    def calcWs(self, alphas, dataArr, classLabels):
        X = np.mat(dataArr)
        labelMat = np.mat(classLabels).transpose()
        m, n = np.shape(X)
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alphas[i] * labelMat[i], X[i, :].T)
        return w

    def predict(self, inx):
        kEval = svm.kernelTrans(self.X[self.sVectorIndex], inx)
        predict = kEval.T * np.multiply(self.labelMat[self.sVectorIndex], self.alphas[self.sVectorIndex]) + self.b
        if predict < 0:
            return -1
        else:
            return 1


def createDataSet():
    iris = load_iris()
    datasets = []
    labels = []
    for i, data in enumerate(iris.data):
        if iris.target[i] == 0:
            continue

        if iris.target[i] == 1:
            labels.append(-1)
            datasets.append([1.0, iris.data[i, 0] + 1.0, iris.data[i, 2] + 0.1])
        else:
            labels.append(1)
            datasets.append([1.0, iris.data[i, 0], iris.data[i, 2]])
    return datasets, labels


dataMatIn, classLabels = createDataSet()
# svm = SMO_SVM(dataMatIn, classLabels, 0.62, 0.001, 3000, ('lin', 1.0))
# # 训练
# b, alphas, sVectorIndex = svm.train()
# print("b :", b)
# # 计算权重
# weights = svm.calcWs(alphas, dataMatIn, classLabels).reshape(np.shape(dataMatIn)[1], )
# print("weights :", weights)
#
# # 获取支持向量及其标签:
# sVectors = np.mat(dataMatIn)[sVectorIndex]
# sVectorLabels = np.mat(classLabels).transpose()[sVectorIndex]
# print("sVectors :", sVectors.shape)
# print("sVectorLabels :", sVectorLabels.shape)

# 预测
errorRateSum = 0
errorCount = 0
testStep = 100
testSize = 100
cs = np.arange(0.5, 0.7, 0.02)
tols = np.arange(0.001, 0.01, 0.001)
ks = np.arange(0.5, 1.5, 0.1)

iter = 0
maxIter = len(cs) * len(tols) * len(ks)
minRate = 1.0
best = []
rateArr = []
plt.figure()
plt.title("Rate Change")

for c in cs:
    for tol in tols:
        for k in ks:
            errorRateSum = 0
            for j in range(testStep):
                # 0.5, 0.004, 0.6
                svm = SMO_SVM(dataMatIn, classLabels, c, tol, 600, ('rbf', k))
                b, alphas, sVectorIndex = svm.train()
                for i in range(testSize):
                    predict = svm.predict(np.mat(dataMatIn[i]))
                    if predict != classLabels[i]:
                        errorCount += 1
                # print("with {} step, error rate:{}".format(j, errorCount / testSize))
                errorRateSum += errorCount / testSize
                errorCount = 0
            currentRate = errorRateSum / testStep
            rateArr.append(currentRate)
            if currentRate < minRate:
                best = [c, tol, k]
                minRate = currentRate
            print("{} currentRate:{},{} minRate:{},{}".format(maxIter - iter, currentRate, [c, tol, k], minRate, best))
            iter += 1

pickle.dump(best, open("best.dat", "wb"))
pickle.dump(rateArr, open("./rate_arr.dat", "wb"))
# colors = "grb"
# for i, data in enumerate(dataMatIn):
#     if i in sVectorIndex:
#         plt.plot(data[1], data[2], "*%s" % colors[classLabels[i] + 1])
#     else:
#         plt.plot(data[1], data[2], ".%s" % colors[classLabels[i] + 1])
# x = np.arange(5.0, 8.0, 0.1)
# y = (-weights[0] - weights[1] * x) / weights[2] - np.array(b)[0][0]

plt.plot(np.arange(0, maxIter, 1), rateArr, "-")
plt.show()
