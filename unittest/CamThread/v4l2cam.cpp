// This is V4L2Camera.cpp
#include "v4l2cam.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
//v4l2 ioctl api返回值为负报错
#define CHECK_V4L2_CALL(call) \ 
    if((call) < 0) { \
        throw std::runtime_error("V4L2 error in " #call ": " + std::string(strerror(errno))); \
    }

V4L2Camera::V4L2Camera(const std::string& device_path, FrameQueue<cv::Mat>* frameQueue)
   : Cam(frameQueue), fd_(-1), streaming_(false) {  // 初始化列表
    fd_ = open(device_path.c_str(), O_RDWR);
    if(fd_ < 0) {
        throw std::runtime_error("Failed to open device: " + device_path);
    }
}

V4L2Camera::~V4L2Camera() {
    Cleanup();
    if(fd_ >= 0) close(fd_);
}

void V4L2Camera::Initialize(int width, int height) {
    // 1. Check capabilities
    v4l2_capability cap{};
    CHECK_V4L2_CALL(ioctl(fd_, VIDIOC_QUERYCAP, &cap));
    
    // 2. Set video format
    v4l2_format fmt{};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG;
    CHECK_V4L2_CALL(ioctl(fd_, VIDIOC_S_FMT, &fmt));

    // 3. Request buffers
    v4l2_requestbuffers req{};
    req.count = 4;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    CHECK_V4L2_CALL(ioctl(fd_, VIDIOC_REQBUFS, &req));

    // 4. Map buffers
    buffers_.resize(req.count);
    for(unsigned i = 0; i < req.count; ++i) {
        v4l2_buffer buf{};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        CHECK_V4L2_CALL(ioctl(fd_, VIDIOC_QUERYBUF, &buf));

        buffers_[i].length = buf.length;
        buffers_[i].start = mmap(nullptr, buf.length, 
            PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);

        if(buffers_[i].start == MAP_FAILED) {
            throw std::runtime_error("mmap failed");
        }

        CHECK_V4L2_CALL(ioctl(fd_, VIDIOC_QBUF, &buf));
    }

    // 5. Start streaming
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    CHECK_V4L2_CALL(ioctl(fd_, VIDIOC_STREAMON, &type));
    streaming_ = true;
}

void* V4L2Camera::CaptureFrame(int width, int height, unsigned int* out_size) {
    if(!streaming_) {
        Initialize(width, height);
    }

    v4l2_buffer buf{};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    
    CHECK_V4L2_CALL(ioctl(fd_, VIDIOC_DQBUF, &buf));
    
    *out_size = buf.bytesused;
    void* frame_data = buffers_[buf.index].start;
    
    // Requeue the buffer
    CHECK_V4L2_CALL(ioctl(fd_, VIDIOC_QBUF, &buf));
    
    return frame_data;
}

// void V4L2Camera::ReleaseFrame(void* frame) {
void ReleaseFrame(void* frame){
    // mmap内存的生命周期由V4L2Camera统一管理
    // 实际释放操作在Cleanup()中统一处理
}

void V4L2Camera::Cleanup() {
    if(streaming_) {
        int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        ioctl(fd_, VIDIOC_STREAMOFF, &type);
        streaming_ = false;
    }

    for(auto& buffer : buffers_) {
        if(buffer.start != MAP_FAILED) {
            munmap(buffer.start, buffer.length);
            buffer.start = MAP_FAILED;
        }
    }
    
    v4l2_requestbuffers req{};
    req.count = 0;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    ioctl(fd_, VIDIOC_REQBUFS, &req);
}

// // Factory implementation 分发器
// std::unique_ptr<Cam> Cam::Create(Type type, const std::string& device_path) {
//     switch(type) {
//         case Type::V4L2:
//             return std::unique_ptr<Cam>(new V4L2Camera(device_path)); // C++11兼容
//         default:
//             throw std::invalid_argument("Unsupported camera type");
//     }
// }
//
void V4L2Camera::Start(int width, int height, const std::string& raw_format) {
    width_ = width;
    height_ = height;
    raw_format_ = raw_format;
    streaming_ = true;
    Thread::start(); 
}

void V4L2Camera::Stop() {
    streaming_ = false;
    Thread::join();
}

void V4L2Camera::run() {
    try {
        Initialize(width_,height_);
        
        while (streaming_) {
            unsigned int frame_size = 0;
            void* frame_data = CaptureFrame(width_, height_, &frame_size);

            if (frame_data && frame_size > 0 && raw_format_ == "raw10") {
                // RAW10转RAW8
                uint16_t* raw10 = static_cast<uint16_t*>(frame_data);
                cv::Mat raw8_mat(height_,width_,CV_8UC1);//先创建cv::Mat，这样的话raw8_mat.data就不需要自己管理了
                // 直接操作Mat的数据指针
                raw10_to_raw8(raw10, raw8_mat.data, width_, height_);
                // 创建共享指针（无需自定义删除器）
                auto frame = std::make_shared<cv::Mat>(raw8_mat);
                
                if (m_frameQueue) {
                    m_frameQueue->addFrame(frame);
                }
            }
            
            ReleaseFrame(frame_data);
        }
    } catch (const std::exception& e) {
        std::cerr << "Capture error: " << e.what() << std::endl;
    }
    Cleanup();
}
