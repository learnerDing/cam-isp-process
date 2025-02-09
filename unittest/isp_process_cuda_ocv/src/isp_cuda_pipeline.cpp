#include "../include/isp_cuda_pipeline.h"

ISPCudaPipeline::ISPCudaPipeline(FrameQueue<cv::Mat>& frameQueue)
    : m_frameQueue(frameQueue), debayerPipeline(nullptr) {}

ISPCudaPipeline::~ISPCudaPipeline() {
    delete debayerPipeline;
}

void ISPCudaPipeline::run() {
    while (true) {
        std::shared_ptr<cv::Mat> frame;
        if (!m_frameQueue.getFrame(frame)) {
            break; // 停止条件
        }

        // 将输入帧转换为 Tensor
        Tensor inputTensor(*frame);//从cv::Mat构造Tensor

        // 执行 Debayer 处理
        if (!debayerPipeline) {//debayerPipeline申明在isp_cuda_pipeline.h文件
            debayerPipeline = new DebayerCudaPipeline(std::move(inputTensor), frame->cols, frame->rows);
        }
        Tensor outputTensor = debayerPipeline->process();

        // 预留其他模块处理
        applyAWB(outputTensor);
        applyCCM(outputTensor);
        applyGamma(outputTensor);

        // TODO: 将结果保存或传递到下一个阶段
    }
}

void ISPCudaPipeline::applyAWB(Tensor& tensor) {
    // TODO: 实现 AWB CUDA 逻辑
}

void ISPCudaPipeline::applyCCM(Tensor& tensor) {
    // TODO: 实现 CCM CUDA 逻辑
}

void ISPCudaPipeline::applyGamma(Tensor& tensor) {
    // TODO: 实现 Gamma CUDA 逻辑
}
