# encoding=utf-8
# Created by Mr.Long on 2017/12/17 0017.
# numpy random模块的基本用法

import numpy as np
import matplotlib.pyplot as plt

# 设置打印时 只显示小数点后两位
np.set_printoptions(precision=2)


def main():
    # 生成取值在0，1区间内的2x3矩阵
    print(np.random.rand(2, 3))

    # 生成取值位在0，10区间内的 2x3矩阵
    print(np.random.randint(0, 10, size=(2, 3)))

    # 正态分布 loc均值,scale标准差
    normal_values = np.random.normal(loc=0, scale=0.1, size=1000)
    plt.subplot(131)
    plt.hist(normal_values, 100, normed=True)
    plt.title("normal")

    # 均匀分布
    uniform_values = np.random.uniform(0, 1, size=1000)
    plt.subplot(132)
    plt.plot(uniform_values, ".r")
    plt.title("uniform")

    # 泊松分布 lam: lanbda系数
    poisson_values = np.random.poisson(lam=2.0, size=1000)
    plt.subplot(133)
    plt.plot(poisson_values, ".g")
    plt.title("poisson")


    # 乱序 - permutation :原数组不变,返回打乱后的数组
    a = np.arange(0, 10, 1)
    print("原始:", a)
    b = np.random.permutation(a)
    print("permutation - 原始:", a)
    print("permutation - 结果:", b)

    # 乱序 - shuffle :直接将原数组打乱
    print("shuffle - 原始:", a)
    np.random.shuffle(a)
    print("shuffle - 结果:", a)

    # 抽取
    a = np.arange(0, 10, 1)
    print("原始:", a.shape)
    # replace:True 可重复抽取,False 不可重复抽取 默认True
    b = np.random.choice(a, size=(2, ), replace=True)
    print("choice:", b)

    plt.show()
    pass


if __name__ == "__main__":
    main()
