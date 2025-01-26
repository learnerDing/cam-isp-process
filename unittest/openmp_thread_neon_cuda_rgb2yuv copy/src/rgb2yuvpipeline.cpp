// this is src/rgb2yuvpipeline.cpp
#include "rgb2yuvpipeline.h"
#include "rgb2yuv.cuh"

RGB2YUVPipeline::RGB2YUVPipeline(Tensor& InTensor_cpu, Tensor& OutTensor_cpu, 
                                int width, int height) 
    : width_(width), height_(height) {
    // 输入检查
    assert(InTensor_cpu.device() == DeviceType::CPU);
    
    // 创建GPU Tensor
    rgb_tensor_gpu_ = InTensor_cpu.cputogpu();
    yuv_tensor_gpu_ = Tensor(InTensor_cpu.shape(), DataType::FLOAT32, DeviceType::GPU);
    
    // 保存引用
    rgb_tensor_cpu = InTensor_cpu;
    yuv_tensor_cpu = OutTensor_cpu;
}

RGB2YUVPipeline::~RGB2YUVPipeline() {
    delete &rgb_tensor_gpu_;
    delete &yuv_tensor_gpu_;
}

Tensor RGB2YUVPipeline::process() {
    // 执行CUDA核函数
    launch_rgb2yuv<float>(
        &rgb_tensor_gpu_,&yuv_tensor_gpu_,
        width_, height_
    );
    
    // 返回CPU Tensor
    return yuv_tensor_gpu_.gputocpu();
}