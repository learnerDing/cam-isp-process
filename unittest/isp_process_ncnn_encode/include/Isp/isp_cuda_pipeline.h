#ifndef ISP_CUDA_PIPELINE_H
#define ISP_CUDA_PIPELINE_H

#include "Tensor.h"
#include "FrameQueue.h"
#include "Thread.h"
#include "debayer/debayer_cuda_pipeline.h"
// #include "awb/awb_cuda_pipeline.h"
// #include "ccm/ccm_cuda_pipeline.h"
// #include "gamma/gamma_cuda_pipeline.h"

class ISPCudaPipeline : public Thread {
public:
    explicit ISPCudaPipeline(FrameQueue<cv::Mat>& frameQueue);
    ~ISPCudaPipeline() override;

protected:
    void run() override;

private:
    FrameQueue<cv::Mat>& m_frameQueue;
    DebayerCudaPipeline* debayerPipeline;

    // 预留其他模块接口
    void applyAWB(Tensor& tensor); // 自动白平衡
    void applyCCM(Tensor& tensor); // 色彩校正矩阵
    void applyGamma(Tensor& tensor); // Gamma 校正
};

#endif // ISP_CUDA_PIPELINE_H
