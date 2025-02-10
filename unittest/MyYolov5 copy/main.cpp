#include "include/InferenceThread.h"
#include "FrameQueue.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>
#include <thread>
#include <fstream>
#define YOLOV5_V62 1  // 根据实际模型版本设置

int main(int argc, char** argv)
{
    if (argc != 2) 
    {
        std::cerr << "Usage: " << argv[0] << " <rgb_file_path>\n";
        return -1;
    }

    // 初始化队列（容量30，满时丢弃最旧的5帧）
    FrameQueue<cv::Mat> frameQueue(30, 5);
    
    // 创建推理线程
    InferenceThread inferenceThread(frameQueue);
    
    // 打开二进制文件
    std::ifstream file(argv[1], std::ios::binary);
    if (!file) 
    {
        std::cerr << "Failed to open RGB file: " << argv[1] << "\n";
        return -1;
    }

    // 视频参数
    const int width = 544;
    const int height = 960;
    const int channels = 3;  // RGB
    const size_t frameSize = width * height * channels;
    const double targetFPS = 29.97;
    const auto frameDelay = std::chrono::milliseconds(static_cast<int>(1000 / targetFPS));

    // 启动推理线程
    inferenceThread.start();

    // 主线程读取二进制数据
    std::vector<uchar> buffer(frameSize);
    auto total_start = std::chrono::high_resolution_clock::now();
    auto nextFrameTime = total_start;

    while (file.read(reinterpret_cast<char*>(buffer.data()), frameSize)) 
    {
        // 等待到下一帧时间
        std::this_thread::sleep_until(nextFrameTime);
        
        // 创建cv::Mat（注意：OpenCV默认使用BGR顺序，这里需要转换）
        cv::Mat rgbFrame(height, width, CV_8UC3, buffer.data());
        cv::Mat bgrFrame;
        cv::cvtColor(rgbFrame, bgrFrame, cv::COLOR_RGB2BGR);
        
        auto frame_ptr = std::make_shared<cv::Mat>(bgrFrame);
        frameQueue.addFrame(frame_ptr);
        
        // 更新下一帧时间
        nextFrameTime += frameDelay;
    }

    // 停止队列并等待推理完成
    frameQueue.stop();
    inferenceThread.join();

    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    
    std::cout << "\n=== Processing Complete ==="
              << "\nTotal processing time: " << total_duration.count() << " seconds\n";

    return 0;
}