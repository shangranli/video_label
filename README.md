# video_label  短视频的分类

# leNet-5 模型，2\*(卷积 + 池化) + 1\*全连接层

1. video_train.py:   训练模型，共两类，{0：cartoon（卡通），1：real（真人视频）}

     模型已经训练好，模型文件在 video_model 路径下，百度网盘地址：https://pan.baidu.com/s/1a-X4oRkTpW29hEOtKhd3HQ ，
     
     关于模型的说明：
     
     该模型共训练了14000步，每张彩色图片的统一尺寸为：54\*96\*3，原尺寸为540\*960\*3，验证的识别率在73%左右，可能原因如下：
     
          1.卡通图像和真人图像在原始尺寸下容易区分，但缩小尺寸后两类图像的特征就会模糊，由于设备原因，为了加快训练速度，
          
            将图像统一缩小10倍是导致识别率较低的主要原因。
           
          2.迭代轮数较少，该模型共迭代了14001论，耗时接近8个小时。
          
          3.网络模型较浅。
           
          解决办法：
           
          1.如果设备允许，在 GPU 上训练模型，增大训练图像尺寸，尺寸缩小2倍或4倍都可以，大于4倍的效果将不理想。
           
          2.增加迭代轮数。
          
          3.增加模型深度

2. make_label.py:   判断一个输入视频为的类别，{0：cartoon（卡通），1：real（真人视频）}

   对于真人视频该模型能较好的区分，而卡通视频的分类效果并不理想，原因如 1 中描述。

     随机截取视频的 NUM_IMAGE 帧作为判断视频标签的依据，NUM_IMAGE 需为奇数，利用训练好的模型该 make_label.py 可直接给出这
     
     NUM_IMAGE帧中属于0 类和 1 类的个数，效果如下：
     
     *******************************************************************
      截取视频帧数： 31     
      
      [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
      
      cartoon:  0 帧   
      
      real:  31 帧   
      
      该视频为真人风格                                                 
     *******************************************************************
     
3. generate_tfrecords.py:   由于训练的图像太多无法同时载入内存训练，所以采用 tfrecord 数据格式保存训练数据。

     所有训练图像在 train_data 路径下，百度网盘地址： https://pan.baidu.com/s/1Z1eN-aYktzY2ei1szfEuaA
     
     训练数据的 tfrecord 在 train_tfrecords 路径下，百度网盘：https://pan.baidu.com/s/1Ce0VZKKW1A9VGiB6lbXgRQ
     
4. video_eval:   验证训练模型效果
      
      测试集在 test_data 路径下，网盘地址：https://pan.baidu.com/s/1gPJCGcsfoNs0kkCw-mYdOA
      
      测试数据的 tfrecord 在test_records 路径下， 网盘地址：https://pan.baidu.com/s/1aLLknevorOIdL-3YqmN9nQ
      
5.  eval_video 文件下为测试模型的短视频
     
     
     



