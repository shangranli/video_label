import tensorflow as tf
import cv2 as cv
import os
import numpy as np


IMAGE_HIGTH = 135
IMAGE_WIDTH = 240
IMAGE_CHANNELS = 3

BATCH_SIZE = 10


def get_path(path):
    path_list = []
    name_list = os.listdir(path)
    for name in name_list:
        path_list.append(path+'\\'+name)
    return name_list, path_list


# 生成整数型属性（图像标签）
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 生成字符串型属性（图像矩阵数据）
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# 生成 tfrecord 模型
def generate_tfrecords(path, tf_path):
    recordfilenum = 0
    bestnum = 1000  # 每个 tfrecord 存放图片的个数
    best_counter = 0  # 第几个图片
    tfrecord_p = tf_path + '\\' + ('train.tfrecord-%.2d' % recordfilenum)
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_p)
    class_name, class_path = get_path(path)

    for index, class_label in enumerate(class_name):
        img_name, img_path = get_path(class_path[index])
        print(class_label)
        for i in range(len(img_path)):
            best_counter += 1
            if best_counter % 100 == 0:
                print("已经处理", best_counter, "个")
            if best_counter % bestnum == 0:
                recordfilenum += 1
                tfrecord_p = tf_path + '\\' + ('train.tfrecord-%.2d' % recordfilenum)
                tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_p)

            img = cv.imread(img_path[i], 1)
            img1 = cv.resize(img, (IMAGE_WIDTH, IMAGE_HIGTH), 0, 0, cv.INTER_LINEAR)
            img1 = img1 / 255.0
            img_raw = img1.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(index),
                'image_raw': _bytes_feature(img_raw)
            }))
            tfrecord_writer.write(example.SerializeToString())
    tfrecord_writer.close()


# 读取tfrecords
def read_tfrecords(path):
    files = tf.train.match_filenames_once(path)
    filename_sqeue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_sqeue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string)
                                       })
    image = tf.decode_raw(features['image_raw'], tf.float64)
    label = features['label']
    imaged = tf.reshape(image, [IMAGE_HIGTH, IMAGE_WIDTH, IMAGE_CHANNELS])
    imaged = imaged * 255.0

    min_after_dequeue = 1000
    capacity = min_after_dequeue + BATCH_SIZE * 3
    image_batch0, label_batch0 = tf.train.shuffle_batch(
        [imaged, label], batch_size=BATCH_SIZE,
        capacity=capacity, min_after_dequeue=min_after_dequeue)

    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        num = 0
        for i in range(10):
            image_batch, label_batch = sess.run([image_batch0, label_batch0])
            label_one_hot = np.zeros((BATCH_SIZE, 2), dtype=float)
            for j in range(len(label_batch)):
                label_one_hot[j, label_batch[j]] = 1.0
                print(label_one_hot[j])
                cv.imwrite(r'l:\ver'+'\\'+str(label_one_hot[j])+'_'+str(num)+'.jpg', image_batch[j])
                num += 1
            # print(sess.run([image,label,high,width]))

        coord.request_stop()
        coord.join(threads)


def main():
    orig_path = r'L:\test_data'
    tfrecord_path = r"L:\test_tfrecords"
    tfrecord_files = r"L:\train_tfrecords\train.tfrecord-*"
    generate_tfrecords(orig_path, tfrecord_path)
    # read_tfrecords(tfrecord_files)


if __name__ == "__main__":
    main()