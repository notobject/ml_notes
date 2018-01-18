# encoding=utf-8
# Created by Mr.Long on 2017/12/28 0028.
# 这是文件的概括
import pygame, sys
import numpy
import random
import cv2

import tensorflow as tf


def conv_input_image():
    # input_shape = None x 240 x 320 x 1
    # y_lable_shape = None x 3
    width, height, channel = 320, 240, 1
    RECTANGLE_SIZE = 10
    BLCOK_MOVE_SPEED = 6
    BRIDGE_SHAPE = (40, 4)
    BRIDGE_MOVE_SPEED = 5
    BLOCK_IMAGE_STEP = 1
    BRIDGE_IMAGE_STAEP = BLOCK_IMAGE_STEP * 12
    success_count = 0
    failed_count = 0

    pygame.init()
    screencaption = pygame.display.set_caption('hello world')
    screen = pygame.display.set_mode([width, height], 0, 8)
    screen.fill([0, 0, 0])
    block_x, block_y = random.randint(0, width - RECTANGLE_SIZE), 0
    tx, ty = -1 if random.randint(0, 100) % 2 == 0 else 1, 1
    bridge_x = width / 2 - BRIDGE_SHAPE[0] / 2
    key_down = True
    key = None
    image_batch = []
    res = numpy.array([0, 0, 0]).reshape([-1, 3])

    x = tf.placeholder(shape=[None, 80, 120, channel], dtype="float32")

    conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1), dtype="float32")
    conv1_b = tf.Variable(tf.constant(shape=[32], value=1.))
    conv1 = tf.nn.conv2d(x, conv1_w, strides=[1, 1, 1, 1], padding="VALID", use_cudnn_on_gpu=True)
    conv1_w_plus_b = tf.nn.bias_add(conv1, conv1_b)
    relu1 = tf.nn.relu(conv1_w_plus_b)
    pool1 = tf.nn.avg_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    # image_1 = tf.minimum(pool1 * 255, 255)
    print("image_1: ", relu1.get_shape())
    # tf.summary.histogram("Weights", conv1_w)
    # tf.summary.histogram("Bais", conv1_b)

    conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1), dtype="float32")
    conv2_b = tf.Variable(tf.constant(shape=[64], value=1.))
    conv2 = tf.nn.conv2d(pool1, conv2_w, strides=[1, 1, 1, 1], padding="VALID", use_cudnn_on_gpu=True)
    conv2_w_plus_b = tf.nn.bias_add(conv2, conv2_b)
    relu2 = tf.nn.relu(conv2_w_plus_b, name="relu_1")
    pool2 = tf.nn.avg_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    # image_2 = tf.minimum(pool2 * 255, 255)
    print("image_2: ", relu2.get_shape())
    # tf.summary.histogram("Weights", conv2_w)
    # tf.summary.histogram("Bais", conv2_b)

    conv3_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], stddev=0.1), dtype="float32")
    conv3_b = tf.Variable(tf.constant(shape=[64], value=1.))
    conv3 = tf.nn.conv2d(pool2, conv3_w, strides=[1, 1, 1, 1], padding="VALID", use_cudnn_on_gpu=True)
    conv3_w_plus_b = tf.nn.bias_add(conv3, conv3_b)
    relu3 = tf.nn.relu(conv3_w_plus_b, name="relu_1")
    pool3 = tf.nn.avg_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    # image_3 = tf.minimum(pool3 * 255, 255)
    print("pool3_3: ", pool3.get_shape())
    reshape3 = tf.reshape(pool3, shape=[-1, 7 * 12 * 64])
    # tf.summary.histogram("Weights", conv3_w)
    # tf.summary.histogram("Bais", conv3_b)

    fc1_w = tf.Variable(tf.truncated_normal(shape=[7 * 12 * 64, 784], stddev=0.1), dtype="float32")
    fc1_b = tf.Variable(tf.constant(shape=[784], value=1.))
    fc_1 = tf.nn.bias_add(tf.matmul(reshape3, fc1_w), fc1_b)
    relu4 = tf.nn.relu(fc_1, name="relu_1")
    # tf.summary.histogram("Weights", fc1_w)
    # tf.summary.histogram("Bais", fc1_b)

    fc2_w = tf.Variable(tf.truncated_normal(shape=[784, 3], stddev=0.1), dtype="float32")
    fc2_b = tf.Variable(tf.constant(shape=[3], value=0.1))
    fc2 = tf.nn.bias_add(tf.matmul(relu4, fc2_w), fc2_b)
    relu5 = tf.nn.relu(fc2, name="relu_1")
    y_out = tf.argmax(relu5)
    print("relu5 - shape:", relu5.get_shape())
    # tf.summary.histogram("Weights", fc2_w)
    # tf.summary.histogram("Bais", fc2_b)

    scount = tf.placeholder("float32")
    fcount = tf.placeholder("float32")
    dr = tf.placeholder("float32", shape=[None, 3])

    cross_entry = -tf.reduce_mean(dr * tf.log(relu5))
    tf.summary.scalar("loss", cross_entry)
    train_setp = tf.train.AdamOptimizer(0.001).minimize(cross_entry)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("./tmp/", sess.graph)
        ckpt = tf.train.get_checkpoint_state("./model/")
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("加载模型...")
        else:
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
        counter = 0
        print("开始训练...")
        while True:

            if key_down:
                if key == pygame.K_LEFT:
                    bridge_x -= BRIDGE_MOVE_SPEED
                    if bridge_x < 0:
                        bridge_x = 0
                elif key == pygame.K_RIGHT:
                    bridge_x += BRIDGE_MOVE_SPEED
                    if bridge_x > width - BRIDGE_SHAPE[0]:
                        bridge_x = width - BRIDGE_SHAPE[0]
            screen.fill([0, 0, 0])
            block_x = block_x + (BLCOK_MOVE_SPEED * tx)
            block_y = block_y + (BLCOK_MOVE_SPEED * ty)
            block_rect = pygame.rect.Rect([block_x, block_y, RECTANGLE_SIZE, RECTANGLE_SIZE])
            brigdge_rect = pygame.rect.Rect([bridge_x, height - BRIDGE_SHAPE[1], BRIDGE_SHAPE[0], BRIDGE_SHAPE[1]])
            screen.fill(color=[255, 255, 255], rect=block_rect)
            screen.fill(color=[255, 255, 255], rect=brigdge_rect)

            if block_rect.colliderect(brigdge_rect):
                success_count += 1
                ty = -1
            elif (block_rect.y >= height - RECTANGLE_SIZE):
                failed_count += 1
                block_x, block_y = random.randint(0, width - RECTANGLE_SIZE), 0
                tx, ty = -1 if random.randint(0, 100) % 2 == 0 else 1, 1
                bridge_x = width / 2 - BRIDGE_SHAPE[0] / 2
            elif (block_rect.x + block_rect.width >= width):
                tx = -1
            elif (block_rect.y <= 0):
                ty = 1
            elif (block_rect.x <= 0):
                tx = 1

            counter += 1
            if counter % 2 != 0:
                continue
            image = pygame.surfarray.pixels2d(screen).reshape([width, height, channel])
            image_batch.append(cv2.resize(image, dsize=(80, 120)))
            for event in pygame.event.get():
                pass
            image_batch = numpy.array(image_batch)
            image_batch = image_batch.reshape([-1, 80, 120, channel]) / 255

            sess.run(train_setp, feed_dict={x: image_batch, dr: res, scount: success_count * 1.0,
                                            fcount: (failed_count + 1) * 1.0})
            if counter != 0 and counter % 1000 == 0:
                saver.save(sess, "./model/game-model", global_step=counter)
            res = sess.run(relu5, feed_dict={x: image_batch})
            direct = sess.run(tf.argmax(res, axis=1), feed_dict={x: image_batch})
            if counter % 2 == 0:
                closs = sess.run(cross_entry,
                                 feed_dict={x: image_batch, dr: res, scount: success_count * 1.0,
                                            fcount: (failed_count + 1) * 1.0})
                print("Step:%d, Sucess:%d, Failed:%d, Loss:%g, Current:" % (
                    counter, success_count, failed_count, closs), res, direct)
            if counter % 10:
                merged_res = sess.run(merged,
                                      feed_dict={x: image_batch, dr: res, scount: success_count * 1.0,
                                                 fcount: (failed_count + 1) * 1.0})
                writer.add_summary(merged_res, global_step=counter)

            image_batch = []
            direction = direct[0]
            if direction == 0:
                bridge_x = bridge_x
            if direction == 1:
                bridge_x -= BRIDGE_MOVE_SPEED
                if bridge_x < 0:
                    bridge_x = 0
            elif direction == 2:
                bridge_x += BRIDGE_MOVE_SPEED
                if bridge_x > width - BRIDGE_SHAPE[0]:
                    bridge_x = width - BRIDGE_SHAPE[0]

            pygame.display.update()
            continue
            pygame.time.delay(10)

    pass


conv_input_image()
pygame.display.flip()
