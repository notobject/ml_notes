# encoding=utf-8
# Created by Mr.Long on 2018/1/17 0017.
# 这是文件的概括

import numpy as np
import matplotlib.pyplot as plt
import math


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    plt.figure()
    x = np.arange(-5, 5, 0.1)
    y = sigmoid(x)
    plt.subplot(211)
    plt.plot(x, y)

    # 当x轴取值范围扩大时,sigmoid函数近似单位阶跃函数
    x = np.arange(-100, 100, 0.1)
    y = sigmoid(x)
    plt.subplot(212)
    plt.plot(x, y)

    plt.show()
    pass


if __name__ == "__main__":
    main()
