// include/rgb2yuvpipeline.h
#pragma once
#include "Tensor.h"

class RGB2YUVPipeline {
public:
    RGB2YUVPipeline(Tensor& InTensor_cpu,Tensor& OutTensor_cpu ,int width, int height) ;
    ~RGB2YUVPipeline();
    
    Tensor process();

private:
    Tensor *d_rgb_, *d_yuv_;
    int width_, height_;
};