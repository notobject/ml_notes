# encoding=utf-8
# Created by Mr.Long on 2017/12/19 0019.
# 这是文件的概括
import matplotlib.pyplot as plt
import numpy as np


def main():
    x = np.random.normal(loc=0, scale=0.1, size=[100])
    noise = np.random.normal(loc=0, scale=0.1, size=[50])
    # 通过改变斜距 将随机生成的数据点 分成两个类别
    y_1 = 2.0 * x[:50] + 0.5 + noise
    y_2 = 2.0 * x[50:] + 1.5 + noise

    plt.figure()
    plt.plot(x[:50], y_1, "+r", label="y = 1")
    plt.plot(x[50:], y_2, ".b", label="y = -1")
    plt.legend(loc='lower right')
    plt.show()
    pass


if __name__ == "__main__":
    main()
