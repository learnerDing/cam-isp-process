// src/rgb2yuvpipeline.cpp
#include "rgb2yuvpipeline.h"
#include "rgb2yuv.cuh"

RGB2YUVPipeline::RGB2YUVPipeline(Tensor& InTensor_cpu,Tensor& OutTensor_cpu ,int width, int height) 
    : width_(width), height_(height) {
    // 分配GPU内存 (CHW格式)
    d_rgb_ = &(InTensor_cpu.to(DeviceType::GPU));
    d_yuv_ = &(OutTensor_cpu.to(DeviceType::GPU));
}

RGB2YUVPipeline::~RGB2YUVPipeline() {
    delete d_rgb_;
    delete d_yuv_;
}

Tensor RGB2YUVPipeline::process() {
    // 数据传输 CPU->GPU
    //Tensor rgb_gpu = rgb_cpu.to(DeviceType::GPU);
    
    // 执行CUDA核函数
    launch_rgb2yuv<float>(d_rgb_, d_yuv_, width_, height_);
    
    // 结果回传 GPU->CPU
    return std::move(d_yuv_->to(DeviceType::CPU));
} 