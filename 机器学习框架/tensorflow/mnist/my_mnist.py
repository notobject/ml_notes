# encoding=utf-8
# Created by Mr.Long on 2017/10/23 0023.
# 用简单的softmax回归模型应用在MNIST上,结果只有91%正确率
# http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html

import input_data
import tensorflow as tf


def main():
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    # mnist.train.images -> [60000,784]
    # mnist.train.labels -> [60000,10]

    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # 计算softmax回归模型
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 计算交叉嫡z作为最小化误差所用的损失函数
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # 用梯度下降算法以0.01的速率最小化交叉嫡
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # 初始化所有的变量
    init = tf.initialize_all_variables()

    # 启动模型并初始化变量
    sess = tf.Session()
    sess.run(init)

    # 开始训练模型,循环训练1000次
    # 随机梯度下降训练
    for i in range(1000):
        # 随机抓取训练数据中的100个批处理数据点
        batch_xs, batch_ys = mnist.train.next_batch(100)
        # 用这些数据点作为参数替换之前的占位符来运行train_step
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    pass


if __name__ == "__main__":
    main()
