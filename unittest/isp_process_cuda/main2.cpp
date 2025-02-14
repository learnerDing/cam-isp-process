// #include "include/isp_cuda_pipeline.h"
#include "include/isp_opencv_pipeline.h"

int main() {
    FrameQueue<cv::Mat> frameQueue(30);

    // 选择使用 CUDA 或 OpenCV 管道
#ifdef USE_CUDA
    ISPCudaPipeline pipeline(frameQueue);
#else
    ISPOpenCVPipeline pipeline(frameQueue);
#endif

    pipeline.start();

    // 模拟向队列中添加帧
    for (int i = 0; i < 10; ++i) {
        cv::Mat rawImage = CVReadraw();
        frameQueue.addFrame(std::make_shared<cv::Mat>(rawImage));
    }

    pipeline.join();
    return 0;
}
