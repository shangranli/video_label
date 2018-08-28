import tensorflow as tf
import time
import video_inference
import cv2 as cv
import numpy as np


BATCH_SIZE = 300    # batch大小
IMAGE_HIGH = 54     # 图像高
IMAGE_WIDTH = 96    # 图像宽
IMAGE_CHANNELS = 3  # 图像通道
OUTPUT_NODES = 2    # 类别数

EVAL_INTERVAL_SECS = 10
MOVING_AVERAGE_DECAY = 0.99

NUM_IMAGE = 31      # 随机截取每个视频的帧数


# 随机截取视频的 NUM_IMAGE 个图像，统一尺寸后放到 data 中，作为神经网络的输入
def process_video(_video_path):
    data = np.empty((NUM_IMAGE, IMAGE_HIGH, IMAGE_WIDTH, IMAGE_CHANNELS))  #初始化网络输入数据
    cap = cv.VideoCapture(_video_path)

    frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print('总帧数：', frame_count)
    fps = cap.get(cv.CAP_PROP_FPS)
    print('帧速：', fps, 'fps')

    random_counter = np.random.random_integers(25 * 5, frame_count - 25 * 4, NUM_IMAGE)  # 去掉开头结尾

    for i in range(len(random_counter)):
        cap.set(cv.CAP_PROP_POS_FRAMES, random_counter[i])  # 读取指定帧
        ret, frame = cap.read()
        img = cv.resize(frame, (IMAGE_WIDTH, IMAGE_HIGH), 0, 0, cv.INTER_LINEAR)  # 统一尺寸
        data[i] = img

    cap.release()
    cv.destroyAllWindows()

    return data


def evaluate(data, model_path):
    x = tf.placeholder(tf.float32,
                       [NUM_IMAGE,
                        IMAGE_HIGH,
                        IMAGE_WIDTH,
                        IMAGE_CHANNELS],
                       name='x-input')

    y = video_inference.inference(x, False, None)

    # 预测标签
    prediction_labels = tf.argmax(y, 1)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)

    with tf.Session() as sess:
        print("---预测模型---")
        ckpt = tf.train.get_checkpoint_state(model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            predict_label = sess.run(prediction_labels, feed_dict={x: data})
            print(predict_label)
            cartoon_label = list(predict_label).count(0)
            real_label = list(predict_label).count(1)
            
            # 打印 label, {0:cartoon, 1: real}
            print("cartoon: ", cartoon_label, '\n', 'real: ', real_label)  
        else:
            print("No checkpoint file found")
            return


def main(argv=None):
    # 训练数据的 tfrecord
    train_recorde_path = r"E:\PychormProjects\video_label\train_tfrecords\train.tfrecord-*"
    # 训练模型路径
    model_path = "video_model/"
    # 视频路径
    video_path = r'L:\ver_video\401.mp4'
    # 获得神经网络的输入
    data = process_video(video_path)
    
    evaluate(data, model_path)


if __name__ == '__main__':
    tf.app.run()
    main()
