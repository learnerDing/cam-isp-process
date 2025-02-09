// This is include/debayerpipeline.h
#ifndef DEBAYER_PIPELINE_H
#define DEBAYER_PIPELINE_H

#include "Tensor.h"
#include <stdexcept>
#include "debayer.cuh" 
class DebayerPipeline {
public:
    // 构造函数，接受一个 CPU Tensor 和图像的宽度、高度
    DebayerPipeline(Tensor&& input_tensor_cpu, int width, int height);

    // 析构函数
    ~DebayerPipeline();

    // 处理函数，执行 Debayer 操作并返回 CPU Tensor
    Tensor process();

private:
    int width_;  // 图像宽度
    int height_; // 图像高度

    Tensor input_tensor_gpu_;  // 输入 Tensor（GPU）
    Tensor output_tensor_gpu_; // 输出 Tensor（GPU）
};

#endif // DEBAYER_PIPELINE_H