// This is main.cpp
#include "pipeline.h"
#include "debayer_pipeline.h"
// 其他ISP步骤头文件...

int main() {
    // 配置参数
    const bool USE_CUDA = true;
    
    // 创建处理流水线
    std::unique_ptr<Pipeline> pipeline = Pipeline::create(USE_CUDA);
    
    // 实时处理循环
    while(true) {
        cv::Mat raw = getRawFrame(); // 获取原始帧
        
        // 执行完整处理流程
        cv::Mat processed = pipeline->process(raw);
        
        // 显示结果
        cv::imshow("Result", processed);
        if(cv::waitKey(1) == 27) break;
    }
    return 0;
}