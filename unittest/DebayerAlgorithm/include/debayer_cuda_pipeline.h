// This is include/debayerpipeline.h
#ifndef DEBAYER_PIPELINE_H
#define DEBAYER_PIPELINE_H

#include "Tensor.h"
#include <stdexcept>
#include "debayer.cuh" 
#include "debayer_pipeline.h"

class DebayerCUDAPipeline: public DebayerPipeline {
public:
    DebayerCUDAPipeline(Tensor&& input_tensor_cpu, int width, int height);
    // 处理函数，执行 Debayer 操作并返回 CPU Tensor
    Tensor debayerprocess() ;

private:
    int width_;  // 图像宽度
    int height_; // 图像高度

    Tensor input_tensor_gpu_;  // 输入 Tensor（GPU）
    Tensor output_tensor_gpu_; // 输出 Tensor（GPU）
};

#endif // DEBAYER_PIPELINE_H