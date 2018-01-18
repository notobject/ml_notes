# encoding=utf-8
import numpy as np


def main():
    # python 的列表
    lst = [[1, 2, 3], [4, 5, 6]]
    print(type(lst))

    # 转换为numpy的ndarray(最基本的数据)
    nplst = np.array(lst, dtype=np.float)
    print(type(nplst))
    # 维数
    print(nplst.shape)
    # 维度
    print(nplst.ndim)
    # array的类型
    print(nplst.dtype)
    # 每个元素的字节
    print(nplst.itemsize)
    # 元素数量
    print(nplst.size)

    # ------------------------------------------各种数组
    print(np.zeros([2, 4]))
    print(np.ones([3, 3]))

    # 随机数(均匀分布)
    print(np.random.rand(3, 3))
    print(np.random.rand())

    # 随机整数 [下界,上界,个数)
    print(np.random.randint(1, 10, 3))

    # 正态分布
    print(np.random.randn(2, 4))
    # -------------------------------------------各种操作
    # 等差数列
    nplst = np.arange(1, 11).reshape(2, 5)
    # 自然指数
    print(np.exp(nplst))
    # 平方
    print(np.exp2(nplst))
    # 开平方
    print(np.sqrt(nplst))
    # sin
    print(np.sin(nplst))
    # log
    print(np.log(nplst))
    # sum
    # -减
    # *乘
    # /除
    # **2平方
    # 追加
    # nplst.concatenate(np.arange())

    print("--------------------------------------------矩阵与线性方程组")
    from numpy.linalg import *

    # 单位矩阵
    print(np.eye(3))
    lst = np.array([[1, 2], [3, 4]])
    # 逆 inv
    print(inv(lst))
    # 转置 transpose
    print(lst.transpose())
    # 行列式 det
    print(det(lst))
    # 特征值 eig
    print(eig(lst))
    # 解方程组
    y = np.array([
        [5], [7]
    ])
    print(solve(lst, y))

    pass


if __name__ == "__main__":
    main()
