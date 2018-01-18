# encoding=utf-8
# Created by Mr.Long on 2017/10/30 0030.
# 这是文件的概括
import glob
from itertools import groupby
from collections import defaultdict
import tensorflow as tf

DATA_SET_FILE_PATH = "D:\\DataSet\\DogImages\\"


def data_set():
    training_dataset = defaultdict(list)
    testing_dataset = defaultdict(list)
    image_filenames = glob.glob(DATA_SET_FILE_PATH + "n02*\\*.jpg")
    # 将文件名分解为品种和相应的文件名,品种对应于文件夹名称
    image_filename_with_breed = map(lambda filename: (filename.split("\\")[3], filename),
                                    image_filenames)

    # 依据品种(上述返回元组的第0个分量)对图像分组
    for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
        # 枚举每个品种的图像,并将大致20%的图像划入测试集
        for i, breed_image in enumerate(breed_images):
            # print(dog_breed, " | ", breed_image)
            if i % 5 == 0:
                testing_dataset[dog_breed].append(breed_image[1])
            else:
                training_dataset[dog_breed].append(breed_image[1])

    breed_training_count = len(training_dataset[dog_breed])
    breed_testing_count = len(testing_dataset[dog_breed])
    assert round(breed_testing_count / (breed_testing_count + breed_training_count), 2) > 0.18, "testing Not enough "
    return training_dataset, testing_dataset


def write_records_file(dataset, record_location):
    """

    :param dateset:
    :param record_location:
    :return:
    """
    writer = None
    current_index = 0
    sess = tf.Session()
    for breed, images_filenames in dataset.items():
        for image_filename in images_filenames:
            if current_index % 100 == 0:
                if writer:
                    writer.close()
                record_filename = "{record_location}-{current_index}.tfrecords".format(record_location=record_location, current_index=current_index)
                writer = tf.python_io.TFRecordWriter(record_filename)
            current_index += 1
            image_file = tf.read_file(image_filename)
            try:
                image = tf.image.decode_jpeg(image_file)
            except:
                print(image_filename)
                continue
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, [250, 151])
            image_bytes = sess.run(tf.cast(resized_image, tf.uint8)).tobytes()
            image_label = breed.encode("utf-8")
            example = tf.train.Example(features=tf.train.Features(
                feature={'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])),
                         'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))}))
            writer.write(example.SerializeToString())
            writer.close()
    pass


def main():
    training_dataset, testing_dataset = data_set()
    write_records_file(training_dataset, "./output/training-images/training-image")
    write_records_file(testing_dataset, "./output/testing-images/testing-image")

    pass


if __name__ == "__main__":
    main()
