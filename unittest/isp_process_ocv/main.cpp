#include "include/isp_opencv_pipeline.h"
#include "rawprovide/Raw8Provider.h"

int main() {
    FrameQueue<cv::Mat> frameQueue(20,2);

    // 创建并启动生产者线程
    Raw8Provider provider(frameQueue);
    provider.start();

    // 创建并启动ISP处理线程
    ISPOpenCVPipeline pipeline(frameQueue);
    pipeline.start();

    // 等待处理线程结束
    pipeline.join();

    // 停止生产者线程并等待其结束
    provider.stop();
    provider.join();

    return 0;
}
