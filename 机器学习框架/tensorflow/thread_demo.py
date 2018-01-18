# encoding=utf-8
# Created by Mr.Long on 2018/1/18 0018.
# 这是文件的概括
import tensorflow as tf

def main():

    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 协同启动的线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动线程运行队列

        # .....

    coord.request_stop()  # 停止所有的线程
    coord.join(threads)


if __name__ == "__main__":
    main()
