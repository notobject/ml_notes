# encoding=utf-8
# Created by Mr.Long on 2017/12/4 0004.
# 这是文件的概括
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from numpy import vstack, array, nan
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from numpy import log1p


def main():
    # 加载 IRIS（鸢尾花）数据集:http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
    iris = load_iris()

    # 包含4个特征（Sepal.Length（花萼长度）、Sepal.Width（花萼宽度）、Petal.Length（花瓣长度）、Petal.Width（花瓣宽度））
    print("原始数据:", iris.data[0])

    # 鸢尾花的分类（Iris Setosa（山鸢尾）、Iris Versicolour（杂色鸢尾），Iris Virginica（维吉尼亚鸢尾））
    # print(iris.target)

    """
    **无量纲化**
        使不同规格的数据转换到同一规格。
        常见的无量纲化方法有标准化和区间缩放法。
        标准化的前提是特征值服从正态分布，标准化后，其转换成标准正态分布。
        区间缩放法利用了边界值信息，将特征的取值区间缩放到某个特点的范围，例如[0, 1]等。
    """
    standardData = StandardScaler().fit_transform(iris.data)
    print("标准化:", standardData[0])

    mmData = MinMaxScaler().fit_transform(iris.data)
    print("区间缩放法:", mmData[0])

    """
    **定量特征二值化**
        定量特征二值化的核心在于设定一个阈值，大于阈值的赋值为1，小于等于阈值的赋值为0
        如一个学生的成绩特征,设定阈值为60,则: 成绩>=60分 赋值1, 成绩<60分赋值0
    """
    # 如设置IRIS数据集中的阈值为3 返回二值化后的数据
    binaryData = Binarizer(3).fit_transform(iris.data)
    print("二值化:", binaryData[0])

    """
    **对定性特征哑编码(One-Hot编码,独热编码)**
        由于IRIS数据集的特征皆为定量特征，故使用其目标值进行哑编码（实际上是不需要的）。
    """
    # iris.target.reshape((-1, 1))将一维的target 转换成 N维1列的2维列表,其中 -1表示行数未知,根据划分的列数自动计算行数
    one_hotTarget = OneHotEncoder().fit_transform(iris.target.reshape((-1, 1)))

    print("Before Oon-Hot:", iris.target[0])
    print("One-Hot:", one_hotTarget[0])

    """
    **缺失值计算**
        缺失值计算，返回值为计算缺失值后的数据
        参数missing_value为缺失值的表示形式，默认为NaN
        参数strategy为缺失值填充方式，默认为mean（均值）
    """
    imputerData = Imputer().fit_transform(vstack((array([nan, nan, nan, nan]), iris.data)))
    print("Before Imputer:", iris.data.shape)
    print("Imputer:", imputerData.shape)

    """
    **数据变换**
        常见的数据变换有基于多项式的、基于指数函数的、基于对数函数的
    """
    # 3个特征度为2(degree默认为2)的多项式变换公式 (x,y,z)=(x, y, z, x*x, x*y, x*z, y*y, y*z, z*z)
    pnfData = PolynomialFeatures().fit_transform(iris.data)
    print("多项式变换:", pnfData[0])

    # 基于单变量函数的数据变换,其中log1p可以是任意单变量函数:如 sqrt,floor....
    functData = FunctionTransformer(log1p).fit_transform(iris.data)
    print("单变量函数变换:", functData[0])

    pass


if __name__ == "__main__":
    main()
