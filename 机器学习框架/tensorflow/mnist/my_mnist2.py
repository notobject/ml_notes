# encoding=utf-8
# Created by Mr.Long on 2017/10/23 0023.
# 卷积神经网络模型应用于MNIST数据集(ReLU神经元)
# http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html

import tensorflow as tf
import input_data


def main(is_train=False):
    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
    # mnist.train.images -> [60000,784]
    # mnist.train.labels -> [60000,10]

    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y_ = tf.placeholder("float", [None, 10])

    # 第一层卷积层
    # 卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目
    W_conv1 = weight_variable([5, 5, 1, 32])

    # 对于每一个输出通道都有一个对应的偏置量
    b_conv1 = bias_variable([32])

    # 把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1)
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # 把x_image和权值向量进行卷积 加上偏置项，然后应用ReLU激活函数
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # 进行max pooling
    h_pool1 = max_pool_2x2(h_conv1)

    # -------------------------------
    # 第二层
    # 为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # -------------------------------
    # 密集连接层
    # 现在，图片尺寸减小到7x7，我们加入一个有1024个神经元的全连接层，用于处理整个图片。
    # 我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # 为了减少过拟合，我们在输出层之前加入dropout
    # 用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # -----------------------------------
    # 输出层(softmax层)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # ----------------------------------------------------------评估模型
    sess = tf.Session()

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if (is_train):
        for i in range(20000):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
            train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        saver.save(sess, 'MNIST_data/model/model.ckpt')  # 保存训练好的模型
    else:
        saver.restore(sess, "MNIST_data/model/model.ckpt")
    print("test accuracy %g" % accuracy.eval(session=sess,
                                             feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    pass


def weight_variable(shape):
    # 这个模型中的权重在初始化时应该加入少量的噪声来打破对称性以及避免0梯度
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    # 用一个较小的正数来初始化偏置项，以避免神经元节点输出恒为0的问题
    initial = tf.constant(0.1, shape=shape)
    return initial


# 我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 我们的池化用简单传统的2x2大小的模板做max pooling。
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


if __name__ == "__main__":
    # test()
    main(is_train=True)
