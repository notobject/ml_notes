# encoding=utf-8
# Created by Mr.Long on 2017/12/4 0004.
# 参考:http://www.cnblogs.com/jasonfreak/p/5448385.html
# feature_selection库

from numpy import array
from scipy.stats import pearsonr
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from minepy import MINE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from LR import LR
from sklearn.ensemble import GradientBoostingClassifier


# 由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5
def mic(x, y):
    m = MINE()
    m.compute_score(x, y)
    return m.mic(), 0.5


def main():
    iris = load_iris()
    print("原始数据:", iris.data[0])

    """
    **方差选择法**
        使用方差选择法，先要计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征
    """
    # 方差选择法，返回值为特征选择后的数据
    # 参数threshold为方差的阈值
    varianceData = VarianceThreshold(threshold=0.3).fit_transform(iris.data)
    print("方差选择法选择的数据:", varianceData[0])

    """
       **相关系数法**
           　使用相关系数法，先要计算各个特征对目标值的相关系数以及相关系数的P值
           pearsonr(x,Y):  皮尔逊相关系数 ,x->第i个样本的第i个特征值.Y->第i个样本的类别
    """
    # 选择K个最好的特征，返回选择特征后的数据
    # 第一个参数为计算评估特征是否好的函数，该函数输入特征矩阵和目标向量，输出二元组（评分，P值）的数组，
    # 数组第i项为第i个特征的评分和P值。在此定义为计算相关系数
    # 第二个参数k为选择的特征个数
    # pData = SelectKBest(lambda X, Y: array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2).fit_transform(iris.data,iris.target)
    # print("相关系数法:", pData[0])

    """
       **卡方检验**
           经典的卡方检验是检验定性自变量对定性因变量的相关性。假设自变量有N种取值，因变量有M种取值，
           考虑自变量等于i且因变量等于j的样本频数的观察值与期望的差距，构建统计量： x*x = 求和(((A-E)*A-E)/E)
           这个统计量的含义简而言之就是自变量对因变量的相关性
    """

    chi2Data = SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)
    print("卡方检验:", chi2Data[0])

    """
       **互信息法**
           经典的互信息也是评价定性自变量对定性因变量的相关性的
    """
    # SelectKBest(lambda X, Y: array(list(map(lambda x: mic(x, Y), X.T))).T, k=2).fit_transform(iris.data, iris.target)

    """
       **递归特征消除法**
           递归消除特征法使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练。
    """
    # estimator 基模型(机器学习中的基本模型,此处是logistic回归模型)
    # n_features_to_select 要选择的特征数
    rfeData = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
    print("递归特征消除法:", rfeData[0])

    """
       **基于惩罚项的特征选择法**
           使用带惩罚项的基模型，除了筛选出特征外，同时也进行了降维。
    """
    embeddedData = SelectFromModel(LogisticRegression(penalty="l1", C=0.1)).fit_transform(iris.data, iris.target)
    print("基于惩罚项的特征选择法:", embeddedData[0])

    # 　L1惩罚项降维的原理在于保留多个对目标值具有同等相关性的特征中的一个，所以没选到的特征不代表不重要。
    # 故，可结合L2惩罚项来优化。具体操作为：若一个特征在L1中的权值为1，
    # 选择在L2中权值差别不大且在L1中权值为0的特征构成同类集合，将这一集合中的特征平分L1中的权值，
    # 故需要构建一个新的逻辑回归模型<LR.py>：
    lrData = SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(iris.data, iris.target)
    print("结合L2惩罚项的优化:", lrData[0])

    """
       ** 基于树模型的特征选择法**
           树模型中GBDT也可用来作为基模型进行特征选择
    """
    SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)

    pass


if __name__ == "__main__":
    main()
