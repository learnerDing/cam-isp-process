// this is v4l2cam.h
#pragma once

#include "Cam.h"
#include "util/Thread.h"  // 添加自定义线程头文件
#include <linux/videodev2.h>
#include <cstring>
#include <vector>
#include <omp.h>
#include <arm_neon.h>


class V4L2Camera final : public Cam, public Thread {
public:
    explicit V4L2Camera(const std::string& device_path, FrameQueue<cv::Mat>* frameQueue);
    ~V4L2Camera() override;

    void Start(int width, int height, const std::string& raw_format) override;
    void Stop() override;

private:
    void run() override;
    void Initialize(int width, int height);
    void Cleanup();
    void* CaptureFrame(int width, int height, unsigned int* out_size);
    struct Buffer { void* start; size_t length; };
    
    int fd_ = -1;
    std::vector<Buffer> buffers_;
    // std::atomic<bool> running_{false};
    bool streaming_ = false;
    int width_;
    int height_;
    std::string raw_format_;
};