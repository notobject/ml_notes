# encoding=utf-8
# Created by Mr.Long on 2017/12/19 0019.
# 训练单隐层神经网络
import tensorflow as tf
import numpy as np


def main():
    print("...")
    # 定义模型
    with tf.name_scope("inputs"):
        x = tf.placeholder(dtype="float32", shape=[None, 2], name="x_input")
        y_ = tf.placeholder(dtype="float32", shape=[None], name="y_input")

    print(np.shape(x), np.shape(y_))
    with tf.name_scope("layer_1"):
        w_1 = tf.Variable(tf.truncated_normal(shape=[2, 3], mean=0, stddev=0.1, dtype="float32"), name="w_1")
        tf.summary.histogram("Layer_1/Weights", w_1)
        b_1 = tf.Variable(tf.constant(value=0.1, dtype="float32", shape=[3]), name="b_1")
        tf.summary.histogram("Layer_1/Bias", b_1)
        wx_plus_b_1 = tf.nn.bias_add(tf.matmul(x, w_1), b_1, name="wx_plus_b_1")
        sigmod_1 = tf.nn.sigmoid(wx_plus_b_1, name="sigmod_1")
        sigmod_1 = tf.reshape(sigmod_1, shape=(1, -1))

    with tf.name_scope("layer_2"):
        w_2 = tf.Variable(tf.truncated_normal(shape=[3, 1], mean=0, stddev=0.1, dtype="float32"), name="w_2")
        tf.summary.histogram("Layer_2/Weights", w_2)
        b_2 = tf.Variable(tf.constant(value=0.1, dtype="float32", shape=[1]), name="b_2")
        tf.summary.histogram("Layer_2/Bias", b_2)
        wx_plus_b_2 = tf.nn.bias_add(tf.matmul(sigmod_1, w_2), b_2, name="wx_plus_b_2")
        sigmod_2 = tf.nn.sigmoid(wx_plus_b_2, name="sigmod_2")

    with tf.name_scope("loss"):
        loss = tf.reduce_mean(tf.squared_difference(sigmod_2, y_), name="losss")
        tf.summary.scalar("loss", loss)

    with tf.name_scope("train"):
        step_train = tf.train.GradientDescentOptimizer(0.01).minimize(loss=loss)

    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./logs/", sess.graph)

    sess.run(tf.global_variables_initializer())
    batch_x = np.array([[1, 1], [1, 0], [0, 0]])
    batch_y = np.array([1, 1, 0])
    for i in range(1000):
        feed_dict = {x: batch_x[np.random.randint(0, 3, size=1)], y_: batch_y[np.random.randint(0, 3, size=1)]}
        if i % 50:
            result = sess.run(merged, feed_dict=feed_dict)
            writer.add_summary(result, i)
        if i % 100 == 0:
            print(sess.run(loss, feed_dict={x: [[0, 1]], y_: [1]}))
        sess.run(step_train,
                 feed_dict=feed_dict)
        pass
    print(sess.run(wx_plus_b_2, feed_dict={x: [[1, 1]]}))


if __name__ == "__main__":
    main()
