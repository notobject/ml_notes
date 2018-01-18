# coding=utf-8
from numpy import *


def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 侮辱性言论, 0 正常言论
    return postingList, classVec


def createVocabList(dataSet):
    # 创建一个不重复的词汇表
    vocabList = set([])
    for document in dataSet:
        vocabList = vocabList | set(document)
    return list(vocabList)


# 词集模型
def setOfWords2Vec(vocabList, inputSet):
    # 根据输入的数据集 返回一个词汇表向量
    returnVec = [0] * len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print(u"the word : {0:s} is not in vocabList".format(word))

    return returnVec


# 词袋模型
def bagOfWords2Vec(vocabList, inputSet):
    # 根据输入的数据集 返回一个词汇表向量
    returnVec = [0] * len(vocabList)

    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word : %s is not in vocabList" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    # trainMatrix 文档矩阵
    # trainCategory 类别向量

    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    # 计算该文档属于侮辱性文档的概率
    pA = sum(trainCategory) / float(numTrainDocs)
    # 则属于非侮辱性文档的概率为
    # pB = 1 - pA
    p0Num = ones(numWords)
    p1Num = ones(numWords)

    trainCategorySet = set(trainCategory)
    p0Denom = float(len(trainCategorySet))
    p1Denom = float(len(trainCategorySet))

    # 统计单词出现次数
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # 计算单词出现概率
    p0Vect = log(p0Num / p0Denom)
    p1Vect = log(p1Num / p1Denom)
    return p0Vect, p1Vect, pA


def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def textParse(bigString):
    import re
    strList = re.split(r'\W*', bigString)
    return [tok.lower() for tok in strList if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        words = textParse(open("email/spam/%d.txt" % i).read())
        docList.append(words)
        fullText.extend(words)
        classList.append(1)
        words = textParse(open("email/ham/%d.txt" % i).read())
        docList.append(words)
        fullText.extend(words)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        # 随机构建测试集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])

    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    # 用随机样本集训练
    p0Vect, p1Vect, pSpam = trainNB0(array(trainMat), array(trainClasses))

    # 用随机测试集测试
    errorCount = 0
    for docIndex in testSet:
        rclass = classifyNB(setOfWords2Vec(vocabList, docList[docIndex]), p0Vect, p1Vect, pSpam)
        if rclass != classList[docIndex]:
            errorCount += 1
            print(classList[docIndex], docList[docIndex])
    print("分类错误率:", float(errorCount) / len(testSet))


def calcMostFreq(vocabList, fullText):
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]


def localWords(feed1, feed0):
    import feedparser
    docList = [];
    classList = [];
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)  # NY is class 1
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  # create vocabulary
    top30Words = calcMostFreq(vocabList, fullText)  # remove top 30 words
    for pairW in top30Words:
        if pairW[0] in vocabList: vocabList.remove(pairW[0])
    trainingSet = range(2 * minLen);
    testSet = []  # create test set
    for i in range(20):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del (trainingSet[randIndex])
    trainMat = [];
    trainClasses = []
    for docIndex in trainingSet:  # train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:  # classify the remaining items
        wordVector = bagOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print
    'the error rate is: ', float(errorCount) / len(testSet)
    return vocabList, p0V, p1V


def getTopWords(ny, sf):
    import operator
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = [];
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0: topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0: topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print
    "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**"
    for item in sortedSF:
        print
        item[0]
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print
    "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**"
    for item in sortedNY:
        print
        item[0]


# import feedparser as fp

# ny = fp.parse("http://newyork.craigslist.org/search/stp?format=rss")
# sf = ny = fp.parse("http://sfbay.craigslist.org/search/stp?format=rss")
