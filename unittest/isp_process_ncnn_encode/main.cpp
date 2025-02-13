// #include "include/isp_cuda_pipeline.h"
#include "include/isp_opencv_pipeline.h"

int main() {
// 初始化阶段
FrameQueue<cv::Mat> rawQueue;      // 输入队列
FrameQueue<cv::Mat> inferQueue;     // 推理队列
FrameQueue<AVFrame> encodeQueue;   // 编码队列
    // 选择使用 CUDA 或 OpenCV 管道
#ifdef USE_CUDA
    ISPCudaPipeline IspPipeline(rawQueue);//cudaISP需要进一步优化
#else
    ISPOpenCVPipeline IspPipeline(rawQueue);
    IspPipeline.addInferOutputQueue(&inferQueue);
    IspPipeline.addEncodeOutputQueue(&encodeQueue);

#endif
    // 启动流水线
    IspPipeline.start();

    // 模拟向队列中添加帧
    for (int i = 0; i < 10; ++i) {
        cv::Mat rawImage = CVReadraw();
        frameQueue.addFrame(std::make_shared<cv::Mat>(rawImage));
    }

    pipeline.join();
    return 0;
}
