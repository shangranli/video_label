import tensorflow as tf
import cv2 as cv
import os
import video_inference
import numpy as np

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

BATCH_SIZE = 64
IMAGE_HIGTH = 135
IMAGE_WIDTH = 240
IMAGE_CHANNELS = 3

OUTPUT_NODES = 2
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 15001
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "video_model_cartoon/"
MODEL_NAME = "video_model_carton"

orig_path = r'\test_tfrecords'
train_recorde_path = r"L:\test_tfrecords\train.tfrecord-*"
# generate_tfrecord(orig_path, train_recorde_path


# 读取 tfrecord 模型
def train(path):
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,
                        IMAGE_HIGTH,
                        IMAGE_WIDTH,
                        IMAGE_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODES], name='y-input')

    # 计算前向传播
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = video_inference.inference(x, True, regularizer)

    # 定义滑动平均类
    global_step = tf.Variable(0, trainable=False)
    variable_average = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_average_op = variable_average.apply(tf.trainable_variables())

    # 计算交叉熵及平均值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算损失函数
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 设置学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                               global_step,
                                               50,
                                               LEARNING_RATE_DECAY,
                                               staircase=True)

    # 定义训练op
    train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_average_op]):
        train_op = tf.no_op(name='train')

    # 定义持久化类
    saver = tf.train.Saver()

    # 读取tfrecords
    files = tf.train.match_filenames_once(path)
    filename_sqeue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_sqeue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                          'label': tf.FixedLenFeature([], tf.int64),
                                          'image_raw': tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['image_raw'], tf.float64)
    label = tf.cast(features['label'], tf.int64)

    imaged = tf.reshape(image, [IMAGE_HIGTH, IMAGE_WIDTH, IMAGE_CHANNELS])

    min_after_dequeue = 10000
    capacity = min_after_dequeue + BATCH_SIZE * 3
    image_batch0, label_batch0 = tf.train.shuffle_batch(
        [imaged, label], batch_size=BATCH_SIZE,
        capacity=capacity, min_after_dequeue=min_after_dequeue)

    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print(sess.run(files))

        for i in range(TRAINING_STEPS):
            label_one_hot = np.zeros((BATCH_SIZE, 2), dtype=float)
            image_batch, label_batch = sess.run([image_batch0, label_batch0])
            for j in range(len(label_batch)):
                label_one_hot[j, label_batch[j]] = 1.0

            _, loss_value, step = sess.run([train_op, loss, global_step],
                                           feed_dict={x: image_batch,
                                                      y_: label_one_hot})

            if i % 10 == 0:
                print("after %d training steps,cross entropy on all data is %g" % (i, loss_value))

            if i % 1000 == 0:
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)

        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    train(train_recorde_path)


if __name__ == '__main__':
    main()