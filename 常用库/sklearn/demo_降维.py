# encoding=utf-8
# Created by Mr.Long on 2017/12/5 0005.
# 常见降维方法
# 1. 基于L1惩罚项的模型降维
# 2. 主成分分析法(PCA)
# 3. 线性判别分析法(LDA)

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main():
    iris = load_iris()
    print("原始数据:", iris.data[0])

    """主成分分析法(PCA)
        PCA是为了让映射后的样本具有最大的发散性,属于无监督降维方法."""
    pcaData = PCA(n_components=2).fit_transform(iris.data)
    print("PCA:", pcaData[0])

    """线性判别分析法(LDA)
            LDA是为了让映射后的样本具有最好的分类性能,属于有监督降维方法."""
    ldaData = LinearDiscriminantAnalysis(n_components=2).fit_transform(iris.data, iris.target)
    print("LDA:", ldaData[0])

    pass


if __name__ == "__main__":
    main()
