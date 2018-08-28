import tensorflow as tf
import time
import video_inference


BATCH_SIZE = 300
IMAGE_HIGH = 54
IMAGE_WIDTH = 96
IMAGE_CHANNELS = 3
OUTPUT_NODES = 2

EVAL_INTERVAL_SECS = 10
MOVING_AVERAGE_DECAY = 0.99


def evaluate(data_path, model_path):
    x = tf.placeholder(tf.float32,
                       [BATCH_SIZE,
                        IMAGE_HIGH,
                        IMAGE_WIDTH,
                        IMAGE_CHANNELS],
                       name='x-input')

    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODES], name='y-input')

    y = video_inference.inference(x, False, None)

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    # 读取tfrecords
    files = tf.train.match_filenames_once(data_path)
    filename_sqeue = tf.train.string_input_producer(files, shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_sqeue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'image_raw': tf.FixedLenFeature([], tf.string),
                                           'image_high': tf.FixedLenFeature([], tf.int64),
                                           'image_width': tf.FixedLenFeature([], tf.int64)
                                       })
    image = tf.decode_raw(features['image_raw'], tf.float64)
    label = tf.cast(features['label'], tf.int64)
    high = tf.cast(features['image_high'], tf.int64)
    width = tf.cast(features['image_width'], tf.int64)

    imaged = tf.reshape(image, [54, 96, 3])

    min_after_dequeue = 1000
    capacity = min_after_dequeue + BATCH_SIZE * 3
    image_batch0, label_batch0 = tf.train.shuffle_batch(
        [imaged, label], batch_size=BATCH_SIZE,
        capacity=capacity, min_after_dequeue=min_after_dequeue)

    with tf.Session() as sess:
        sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(10):
            one_hot_label_batch = []
            image_batch, label_batch = sess.run([image_batch0, label_batch0])
            print("---生成数据---")
            for j in range(len(label_batch)):
                label_one_hot = [0.0] * 2
                label_one_hot[label_batch[j]] = 1.0
                one_hot_label_batch.append(label_one_hot)

            print("---预测模型---")
            ckpt = tf.train.get_checkpoint_state(model_path)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                accuracy_score = sess.run(accuracy, feed_dict={x: image_batch, y_: one_hot_label_batch})
                print("After %s training steps, validation accuracy = %g" % (global_step, accuracy_score))
            else:
                print("No checkpoint file found")
                return
            time.sleep(EVAL_INTERVAL_SECS)
        coord.request_stop()
        coord.join(threads)


def main(argv=None):
    train_recorde_path = r"E:\PychormProjects\video_label\test_tfrecords\test.tfrecord-*"
    model_path = "video_model/"
    evaluate(train_recorde_path, model_path)


if __name__ == '__main__':
    tf.app.run()
    main()