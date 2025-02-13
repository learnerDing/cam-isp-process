// this is v4l2cam.h
#pragma once

#include "Cam.h"
#include <linux/videodev2.h>
#include <cstring>
#include <vector>
#include <omp.h>
#include <arm_neon.h>


class V4L2Camera final : public Cam {
public:
    explicit V4L2Camera(const std::string& device_path);
    ~V4L2Camera() override;

    void* CaptureFrame(int width, int height, unsigned int* out_size) override;
    void ReleaseFrame(void* frame) override;

private:
    struct Buffer {
        void*  start;
        size_t length;
    };

    void Initialize(int width, int height);
    void Cleanup();
    
    int fd_ = -1;
    std::vector<Buffer> buffers_;
    bool streaming_ = false;
};