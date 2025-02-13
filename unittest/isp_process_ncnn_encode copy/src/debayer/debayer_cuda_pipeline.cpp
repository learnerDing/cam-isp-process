// this is src/debayer/debayer_cuda_pipeline.cpp
#include "debayer_cuda_pipeline.h"
 // 包含 CUDA 核函数声明

#define DBG_DEBAYER

DebayerCudaPipeline::DebayerCudaPipeline(Tensor&& input_tensor_cpu, int width, int height)
    : width_(width), height_(height) {
    // 输入检查
    assert(input_tensor_cpu.device() == DeviceType::CPU);
    assert(input_tensor_cpu.shape().size() == 3 && input_tensor_cpu.shape()[0] == 1 && "Input must be CHW [1, H, W]");

    // 将输入 Tensor 从 CPU 转移到 GPU
    input_tensor_gpu_ = input_tensor_cpu.cputogpu();

    // 创建输出 Tensor（GPU），形状为 [3, H, W]
    std::vector<int> output_shape = {3, height, width};
    output_tensor_gpu_ = Tensor(output_shape, DataType::FLOAT32, DeviceType::GPU);
}

DebayerCudaPipeline::~DebayerCudaPipeline() {
#ifdef DBG_DEBAYER
    printf("~DebayerCudaPipeline()\n");
#endif
}

Tensor DebayerCudaPipeline::process() {
    // 执行 CUDA 核函数
    input_tensor_gpu_.print(0, 540, 980,1280, "input_tensor_gpu_ before launch cuda print");
    launchDebayer(input_tensor_gpu_, output_tensor_gpu_);

    output_tensor_gpu_.print(0, 540, 980,1280, "output_tensor_gpu_ after launch cuda print");
    //output_tensor_gpu_.print(1, 540, 980,1280, "output_tensor_gpu_ after launch cuda print");
    //output_tensor_gpu_.print(2, 540, 980,1280, "output_tensor_gpu_ after launch cuda print");
    // 返回 CPU Tensor，右值引用不拷贝
    return output_tensor_gpu_.gputocpu();
}