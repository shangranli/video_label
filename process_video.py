import cv2 as cv
import os
import numpy as np

video_path = r'E:\my_video1'
image_path = r'E:\my_image'

def get_path(path):

    path_list = []
    fname_list = os.listdir(path)
    name_list = []
    for name in fname_list:
        path_list.append(path+'\\'+name)
        name_list.append(name.split('.mp4')[0])
    return name_list, path_list


if __name__=="__main__":
    name_list, path_list = get_path(video_path)
    for i in range(len(path_list)):
        print('正在处理第', i, '个', '共', len(path_list))
        cap = cv.VideoCapture(path_list[i])

        frame_count = cap.get(cv.CAP_PROP_FRAME_COUNT)
        print('总帧数：', frame_count)
        fps = cap.get(cv.CAP_PROP_FPS)
        print('帧速：', fps, 'fps')

        random_counter = np.random.random_integers(25*5, frame_count-25*4, 30)  # 去掉开头结尾
        #print(random_counter)

        for random_coun in random_counter:
            cap.set(cv.CAP_PROP_POS_FRAMES, random_coun)  # 读取指定帧
            ret, frame = cap.read()
            #gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  # 灰度转换

            #img = cv.resize(frame, (200, 200), 0, 0, cv.INTER_LINEAR)  # 统一尺寸
            '''cv.imshow('fram',frame)
            cv.waitKey(1000)'''
            path = image_path + '\\' + name_list[i] + '_' + str(random_coun) + '.jpg'
            cv.imwrite(path, frame)
        cap.release()
        cv.destroyAllWindows()





