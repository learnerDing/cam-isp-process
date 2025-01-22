#include "CamThread.h"
#include "FrameQueue.h"
#include <iostream>
#include <cstdlib>  // for std::atoi
//#include <opencv2/opencv.hpp>
// 默认参数
const int DEFAULT_WIDTH = 1920;
const int DEFAULT_HEIGHT = 1080;
const int DEFAULT_RAWBIT = 10;

int main(int argc, char* argv[]) {
    // 解析命令行参数
    int width = DEFAULT_WIDTH;
    int height = DEFAULT_HEIGHT;
    int rawbit = DEFAULT_RAWBIT;

    if (argc >= 2) {
        width = std::atoi(argv[1]);  // 第一个参数是宽度
    }
    if (argc >= 3) {
        height = std::atoi(argv[2]);  // 第二个参数是高度
    }
    if (argc >= 4) {
        rawbit = std::atoi(argv[3]);  // 第三个参数是 rawbit
    }

    // 检查 rawbit 是否合法
    if (rawbit != 8 && rawbit != 10) {
        std::cerr << "Error: rawbit must be 8 or 10." << std::endl;
        return EXIT_FAILURE;
    }

    // 打印参数信息
    std::cout << "Using parameters: width=" << width << ", height=" << height << ", rawbit=" << rawbit << std::endl;

    // 创建帧队列
    FrameQueue<std::shared_ptr<cv::Mat>> frameQueue(10);  // 队列最大容量为 10

    // 创建摄像头线程
    CamThread camThread(frameQueue, "/dev/video0", width, height, rawbit);

    // 启动摄像头线程
    camThread.start();

    // 主线程等待摄像头线程结束
    camThread.join();

    return EXIT_SUCCESS;
}