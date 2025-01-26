// This is include/rgb2yuvpipeline.h
#pragma once
#include "Tensor.h"

class RGB2YUVPipeline {
public:
    // RGB2YUVPipeline(Tensor* InTensor_cpu,Tensor* OutTensor_cpu ,int width, int height) ;
   RGB2YUVPipeline(Tensor& InTensor_cpu, Tensor& OutTensor_cpu, 
                                int width, int height);
    ~RGB2YUVPipeline();
    
    Tensor process();
    Tensor rgb_tensor_cpu,yuv_tensor_cpu;
private:
    Tensor rgb_tensor_gpu_, yuv_tensor_gpu_;
    int width_, height_;
};