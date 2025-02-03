#include "include/FrameQueue.h"
#include "include/InferenceThread.h"
#include <opencv2/opencv.hpp>
#include <memory>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s [videopath]\n", argv[0]);
        return -1;
    }

    const char* videopath = argv[1];
    cv::VideoCapture cap(videopath);
    if (!cap.isOpened()) {
        fprintf(stderr, "Failed to open video %s\n", videopath);
        return -1;
    }

    FrameQueue<cv::Mat> frameQueue(30); // 创建帧队列，容量30
    InferenceThread inferenceThread(&frameQueue);
    inferenceThread.start(); // 启动推理线程

    cv::Mat frame;
    while (cap.read(frame)) { // 读取视频帧
        auto framePtr = std::make_shared<cv::Mat>(frame.clone());
        frameQueue.addFrame(framePtr); // 添加帧到队列
    }

    frameQueue.stop(); // 停止队列，通知推理线程结束
    inferenceThread.join(); // 等待推理线程完成

    return 0;
}