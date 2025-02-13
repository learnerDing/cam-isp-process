// #include "include/isp_cuda_pipeline.h"
#include "include/Isp/isp_opencv_pipeline.h"
#include "include/Cam/Cam.h"
#include "include/Inference/InferenceThread.h"
#include "include/Inference/PreviewThread.h"
#include "include/Encode/EncodeThread.h"

int main() {
// 初始化阶段，初始化缓冲区队列
FrameQueue<cv::Mat> rawQueue;      // 输入队列
FrameQueue<cv::Mat> inferQueue;     // 推理队列
FrameQueue<AVFrame> encodeQueue;   // 编码队列
//创建各种对象、线程
    //创建配置Cam对象

    // ISP选择使用 CUDA 或 OpenCV 管道
#ifdef USE_CUDA
    ISPCudaPipeline IspPipeline(rawQueue);//cudaISP需要进一步优化
#else
    ISPOpenCVPipeline IspPipeline(rawQueue);
    IspPipeline.addInferOutputQueue(&inferQueue);
    IspPipeline.addEncodeOutputQueue(&encodeQueue);
#endif
    //创建配置推理线程


    //创建配置编码线程

    // 启动流水线
    IspPipeline.start();




    return 0;
}
