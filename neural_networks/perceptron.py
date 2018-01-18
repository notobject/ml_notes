# encoding=utf-8
# Created by Mr.Long on 2017/12/19 0019.
# 两个输入神经元的感知机 实现与或非运算

import tensorflow as tf
import numpy as np


def active(x):
    if x[0][0] >= 0:
        return 1
    else:
        return 0


def main():
    """
    与: x1 = 1 且 x2 = 1, w1 = w2 = 1,       threshold = -2         y -> 1
    或: x1 = 1 或 x2 = 1, w1 = w2 = 1,       threshold = -0.5       y -> 1
    非: x1 = 1          , w1 = -0.6 w2 = 0,  threshold = 0.5        y -> 0
        x1 = 0          , w1 = -0.6 w2 = 0,  threshold = 0.5        y -> 1
    :return:
    """
    # 输入神经元
    x = tf.constant([[1., 0.]])
    # 阈值
    threshold = tf.constant([0.5])
    # 权重
    w = tf.constant([[-0.6], [0]])
    print(np.shape(x))
    print(np.shape(w))

    # 连接
    y = tf.matmul(x, w) + threshold
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # active这里的激活函数是自己定义的阶跃函数
    print(active(sess.run(y)))
    pass


if __name__ == "__main__":
    main()
