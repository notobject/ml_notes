# encoding=utf-8
# Created by Mr.Long on 2017/12/18 0018.
# 这是文件的概括

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def main():
    x = tf.placeholder(dtype="float32", shape=[None, 100])
    y = tf.placeholder(dtype="float32", shape=[None, 1])

    w = tf.Variable(np.random.poisson(lam=2.0, size=[100, 1]), dtype="float32")
    b = tf.Variable(tf.constant(shape=[1], value=0.1), dtype="float32")
    tf.add_to_collection('vars', w)
    tf.add_to_collection('vars', b)

    y_ = tf.nn.bias_add(tf.matmul(x, w), b)
    tf.add_to_collection('vars', y_)
    # 均方误差
    mse = tf.reduce_mean(tf.squared_difference(y_, y))
    # L2范式
    l2 = tf.reduce_mean(tf.square(w))
    loss = mse + 0.15 * l2

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss=loss)
    tf.train.Optimizer
    DataSetNum = 5000
    datas_x, datas_y = make_regression(DataSetNum)
    x_train, x_test, y_train, y_test = train_test_split(datas_x, datas_y, test_size=0.2)
    print(x_train.shape)
    print(y_train.shape)
    for i in range(1000):
        batch_rand = np.random.randint(0, DataSetNum * (1 - 0.2), size=50)
        batch_x = np.array([datas_x[i] for i in batch_rand])
        batch_y = np.array([datas_y[i] for i in batch_rand])
        batch_y = batch_y.reshape(-1, 1)
        if i % 100 == 0:
            print("loss:", sess.run(loss, feed_dict={x: batch_x, y: batch_y}))
            # all_vars = tf.get_collection('vars')
            # for v in all_vars:

        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
    print("final loss:", sess.run(loss, feed_dict={x: x_test, y: y_test.reshape(-1, 1)}))
    w_v = sess.run(w)
    b_v = sess.run(b)
    print(w_v.shape)
    print(b_v.shape)
    plt.subplot("121")

    plt.plot(x_train, y_train, ".r")
    plt.subplot("122")
    plt.plot(w_v + b_v, w_v, "+g")
    plt.show()
    pass


def main2():
    # y = 2.0x + 0.5 + noise
    x = np.random.normal(loc=0, scale=0.1, size=100)
    noise = np.random.normal(loc=0, scale=0.1, size=100)
    y = 2.0 * x + 0.5 + noise

    x_input = tf.placeholder(dtype="float32", shape=[None, 1])
    y_input = tf.placeholder(dtype="float32", shape=[1])

    w = tf.Variable(tf.constant(0., shape=[1, 1]))
    b = tf.Variable(tf.constant(0., shape=[1]))
    y_ = tf.matmul(x_input, w) + b

    loss = tf.reduce_mean(tf.squared_difference(y_, y_input))

    train_step = tf.train.GradientDescentOptimizer(0.10).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        index = np.random.randint(0, 100)
        if i % 100 == 0:
            loss_value = sess.run(loss, feed_dict={x_input: x[index].reshape(1, -1), y_input: y[index].reshape(1, )})
            print("train %d ,loss:%g " % (i, loss_value))
        sess.run(train_step, feed_dict={x_input: x[index].reshape(1, -1), y_input: y[index].reshape(1, )})

    plt.figure()
    # 原始数据分布
    plt.plot(x, y, '.r')

    wv, bv = sess.run([w, b])
    print("w:", wv[0][0])
    print("b:", bv[0])
    # 用训练得到的k和b生成点集 画出拟合的直线
    # 形似 y =  2.0 * x +  0.5
    y = wv[0][0] * x + bv[0]
    plt.plot(x, y, '-b')
    plt.show()

    pass


if __name__ == "__main__":
    main2()
