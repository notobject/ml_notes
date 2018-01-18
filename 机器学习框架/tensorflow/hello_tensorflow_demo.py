# encoding=utf-8
# Created by Mr.Long on 2017/10/23 0023.
# 这是文件的概括

import tensorflow as tf


def main():
    hello = tf.constant('Hello, TensorFlow')
    sess = tf.Session()
    print(sess.run(hello))
    a = tf.constant(10)
    b = tf.constant(32)
    print(sess.run(a + b))

    sess.close()
    pass


def load_image(filename):
    file_queue = tf.train.string_input_producer([filename])  # 创建输入队列
    image_reader = tf.WholeFileReader()
    _, image = image_reader.read(file_queue)
    image = tf.image.decode_jpeg(image)
    return image


def main():
    image = load_image("")
    with tf.Session() as sess:
        coord = tf.train.Coordinator()  # 协同启动的线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)  # 启动线程运行队列

        image_loaded = sess.run(image)

        # 从中心位置开始裁剪出图像的10%（0.1）
        central_crop = tf.image.central_crop(image_loaded, 0.1)
        sess.run(central_crop)

        # 当背景有用时 随机化剪裁区域起始位置到图像中心的偏移量来实现裁剪
        bounding_crop = tf.image.crop_to_bounding_box(image_loaded, offset_height=0, offset_width=0, target_height=2,
                                                      target_width=1)
        sess.run(bounding_crop)

        # 边界填充
        pad = tf.image.pad_to_bounding_box(image_loaded, offset_height=0, offset_width=0, target_height=4,
                                           target_width=4)
        sess.run(pad)

        # 剪裁和填充组合(用于处理长宽比不一致的图像数据集)
        tf.image.resize_image_with_crop_or_pad(image_loaded, target_height=2, target_width=2)

        # 图像翻转(这两个函数可对张量进行操作,不仅限于图像)
        # 先对图像进行水平翻转
        flip_horizon = tf.image.flip_left_right(image_loaded)
        # 再垂直翻转
        flip_vertical = tf.image.flip_up_down(flip_horizon)
        sess.run(flip_vertical)

        # 随机翻转
        # tf.image.random_flip_left_right(image_loaded)

        # 饱和与平衡
        # 增加图像的灰度值(增加了0.2)
        adjust_brightness = tf.image.adjust_brightness(image_loaded, 0.2)
        sess.run(adjust_brightness)

        # 调整图像的对比度(降低了0.5)
        adjust_contrast = tf.image.adjust_contrast(image_loaded, -0.5)
        sess.run(adjust_contrast)

        # 调整图像的色度(使图像色彩更为丰富) 增加了0.7
        adjust_hue = tf.image.adjust_hue(image_loaded, 0.7)
        sess.run(adjust_hue)

        # 调整图像饱和度(增加饱和度能够突出颜色的变化,是边缘检测处理的常见的手段)
        adjust_saturation = tf.image.adjust_saturation(image_loaded, 0.4)
        sess.run(adjust_saturation)

        # 颜色
        # 将图像灰度化
        gray = tf.image.rgb_to_grayscale(image_loaded)
        sess.run(gray)

        # HSV空间
        hsv = tf.image.rgb_to_hsv(tf.image.convert_image_dtype(image_loaded, dtype=tf.float32))
        sess.run(hsv)

        # RGB空间
        # LAB空间 (tensorflow没有提供支持,可以选择python-colormath 库进行处理)


        # 从TFRecord文件中加载数据和标签
        # image_height, image_width, image_channels = write_TFRecord_file(image_loaded,
        #                                                                 './output/training-image.tfrecord', image_label)
        #
        # tf_record_image, tf_record_label = load_TFRecord_file('./output/training-image.tfrecord', image_height,
        #                                                       image_width,
        #                                                       image_channels)
        # sess.run(tf.equal(image, tf_record_image))
        # sess.run(image_labels)
        coord.request_stop()  # 停止所有的线程
        coord.join(threads)

if __name__ == "__main__":
    main()
