// This is src/debayer_pipeline.cpp
#include "../include/debayer_pipeline.h"
#include "../include/debayer_cuda_pipeline.h" // CUDA 实现
#include "../include/debayer_ocv_pipeline.h"  // OpenCV 实现
//工程模式分发
// std::unique_ptr<DebayerPipeline> DebayerPipeline::create(bool useCuda, Tensor&& input, int w, int h) {
//     if (useCuda) {
//         return std::make_unique<DebayerCUDAPipeline>(std::move(input), w, h); // 传递参数 // 返回 CUDA 实现
//     } else {
//         return std::make_unique<DebayerOCVPipeline>();  // 返回 OpenCV 实现
//     }
// }
//opencv实现
std::unique_ptr<DebayerPipeline> DebayerPipeline::create(int w, int h) 
{       
    // assert(useCuda==0&& "Use cuda but no Tensor!");
    return std::make_unique<DebayerOCVPipeline>();  // 返回 OpenCV 实现
}
//cuda实现
std::unique_ptr<DebayerPipeline> DebayerPipeline::create(bool useCuda, Tensor&& input,int w, int h) 
{       
    assert(useCuda==1&& "Never use cuda!");
    return std::make_unique<DebayerOCVPipeline>();  // 返回 CUDA 实现
}