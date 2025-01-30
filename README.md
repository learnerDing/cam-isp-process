# CamIspProcess

#### 介绍

开源流媒体处理框架
#### 软件架构
cam-isp-process
1、使用标准v4l2接口采集图像，原始图像支持raw8或者raw10格式，放入缓冲区队列1中。

2、缓冲区队列FrameQueue设计，使用模板支持队列中的frame为cv::Mat和ffmpeg中的AVFrame，定义出队入队函数使用深拷贝保证线程安全。

3、isp pipeline设计，采集出的v4l2图像放入缓冲区队列1后（深拷贝），将图像经过解拜耳、自动白平衡、CCM、Gamma处理后送入缓冲区队列2。isp过程可支持openmp,neon,cuda加速。

4、如果开启cuda加速进行isp，则从缓冲区队列1取出cv::Mat类型（深拷贝）转化为Tensor类型，经过解拜耳、自动白平衡、CCM、Gamma后转化为cv::Mat送入缓冲区队列2。

5、以上v4l2+isp pipeline作为生产者，可以对接以下消费者。

6、消费者1：ffmpeg编码线程，缓冲区队列2出队图像经过rgb转yuv后送入编码线程，将图像帧编成h264视频存放在本地。（支持jetson的nvmpi硬件编码器）

7、消费者2：渲染线程，缓冲区队列2出队图像经过rgb转yuv后送入渲染线程，将yuv图像通过x服务器和opengl渲染到屏幕输出。




#### 安装教程

tobedone
#### 使用说明
tobedone

#### 参与贡献

