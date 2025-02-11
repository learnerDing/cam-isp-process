#include "include/InferenceThread.h"
#include "FrameQueue.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>
#include <thread>
#include <fstream>
#include "PreviewThread.h"
#define YOLOV5_V62 1  // 根据实际模型版本设置



int main(int argc, char** argv)
{
    const bool ENABLE_PREVIEW = true; // 预览开关

    FrameQueue<cv::Mat> frameQueue(30, 5);
    FrameQueue<cv::Mat> previewQueue(3, 1); // 预览专用队列
    //推理线程如果开启预览，对接两个队列
    //frameQueue传递队列本身，previewQueue因为可能为空所以传递指针
    InferenceThread inferenceThread(frameQueue, 
                                  ENABLE_PREVIEW ? &previewQueue : nullptr);
    PreviewThread previewThread(previewQueue);
    if (ENABLE_PREVIEW) {
        previewThread.start();//如果打开预览开关，则启动预览线程
    }
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
    previewQueue.stop();
    inferenceThread.join();
    if (ENABLE_PREVIEW) {
        previewThread.join();
    }
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    
    std::cout << "\n=== Processing Complete ==="
              << "\nTotal processing time: " << total_duration.count() << " seconds\n";

    return 0;
}