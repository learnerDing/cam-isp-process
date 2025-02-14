//This is yuv_encode.cpp
// This is yuv_encode.cpp
#include "include/FrameQueue.h"
#include "include/EncodeThread.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
// #include <chrono>   // 添加chrono头文件
// #include <thread>   // 添加thread头文件

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] 
                  << " <output_base> <codec> <yuv_file> <width> <height>\n";
        return 1;
    }

    // 初始化参数
    const std::string outputBase = argv[1];
    const std::string codecName = argv[2];
    const std::string yuvFile = argv[3];
    const int width = std::stoi(argv[4]);
    const int height = std::stoi(argv[5]);

    // 打开 YUV 文件
    std::ifstream yuvStream(yuvFile, std::ios::binary);
    if (!yuvStream) {
        std::cerr << "Error opening YUV file\n";
        return 1;
    }

    // 初始化帧队列和编码线程
    FrameQueue<AVFrame> frameQueue(10,1);
    EncodeThread encoder(frameQueue, codecName, width, height, outputBase);
    encoder.start();

    // 准备 AVFrame
    AVFrame* frame = av_frame_alloc();
    frame->format = AV_PIX_FMT_YUV420P;
    frame->width = width;
    frame->height = height;
    av_frame_get_buffer(frame, 0);

    // 循环读取 YUV 数据
    const size_t ySize = width * height;
    const size_t uvSize = ySize / 4;
    int64_t pts = 0;
    
    // 使用C++11标准的持续时间定义
    const auto FRAME_INTERVAL = std::chrono::microseconds(33333); // 33.333毫秒

    while (true) {
        auto frame_start = std::chrono::steady_clock::now();
        bool frame_valid = true;

        // 读取 Y 分量
        yuvStream.read(reinterpret_cast<char*>(frame->data[0]), ySize);
        if (yuvStream.eof()) {
            yuvStream.clear();
            yuvStream.seekg(0);
            frame_valid = false;
        }

        if (frame_valid) {
            // 读取 UV 分量
            yuvStream.read(reinterpret_cast<char*>(frame->data[1]), uvSize);
            yuvStream.read(reinterpret_cast<char*>(frame->data[2]), uvSize);
            
            // 检查UV分量是否读取完整
            if (yuvStream.gcount() != uvSize) {
                yuvStream.clear();
                yuvStream.seekg(0);
                frame_valid = false;
            }
        }

        if (frame_valid) {
            frame->pts = pts++;
            frameQueue.addFrame(std::shared_ptr<AVFrame>(
                av_frame_clone(frame),
                [](AVFrame* f) { av_frame_free(&f); }
            ));
        }

        // 精确控制帧率
        auto frame_end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            frame_end - frame_start);
        auto sleep_time = FRAME_INTERVAL - elapsed;
        
        if (sleep_time > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleep_time);
        }
    }

    // 清理资源
    av_frame_free(&frame);
    frameQueue.stop();
    encoder.join();
    return 0;
}
