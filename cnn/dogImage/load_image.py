# encoding=utf-8
# Created by Mr.Long on 2017/10/27 0027.
# 这是文件的概括
import tensorflow as tf


# 加载图像
def load_image(filename):
    file_queue = tf.train.string_input_producer([filename])  # 创建输入队列
    image_reader = tf.WholeFileReader()
    _, image = image_reader.read(file_queue)
    image = tf.image.decode_jpeg(image)
    return image


# 将张量类型的图像保存为TFRecord格式
def write_TFRecord_file(image_loaded, filename, image_label):
    image_bytes = image_loaded.tobytes()
    image_height, image_width, image_channels = image_loaded.shape

    # 导出TFRecord
    writer = tf.python_io.TFRecordWriter(filename)

    # 在样本文件中不保存图像的高度,宽度或通道数,以便节省不要求分配的空间
    example = tf.train.Example(features=tf.train.Features(feature={
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))}))
    writer.write(example.SerializeToString())
    writer.close()
    return image_height, image_width, image_channels


def load_TFRecord_file(filename, image_height, image_width, image_channels):
    file_queue = tf.train.string_input_producer([filename])  # 创建输入队列
    tf_record_reader = tf.TFRecordReader()
    _, tf_record_serialized = tf_record_reader.read(file_queue)

    tf_record_features = tf.parse_single_example(tf_record_serialized, features={
        'label': tf.FixedLenFeature([], tf.string),
        'image': tf.FixedLenFeature([], tf.string),
    })
    tf_record_image = tf.decode_raw(tf_record_features['image'], tf.uint8)
    tf_record_image = tf.reshape(tf_record_image, [image_height, image_width, image_channels])
    tf_record_label = tf.cast(tf_record_features['label'], tf.string)
    return tf_record_image, tf_record_label


def main():
    path = './5.jpg'
    image = load_image(path)
    image_label = b'\0x01'
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
