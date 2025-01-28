// This is include/bgr2yuvpipeline.h
#pragma once
#include "Tensor.h"
#define DBG_BGR2YUV
class BGR2YUVPipeline {
public:
    // BGR2YUVPipeline(Tensor* InTensor_cpu,Tensor* OutTensor_cpu ,int width, int height) ;
   BGR2YUVPipeline(Tensor&& InTensor_cpu, //接受Tensor类型的右值
                                int width, int height);
    ~BGR2YUVPipeline();
    
    Tensor process();
    // Tensor bgr_tensor_cpu,yuv_tensor_cpu;
private:
    Tensor bgr_tensor_gpu_, yuv_tensor_gpu_;
    int width_, height_;
};