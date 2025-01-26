// This is main.cpp
#include <opencv2/opencv.hpp>
#include "rgb2yuvpipeline.h"

// main.cpp
int main() {
    cv::Mat rgb_mat = cv::imread("input.jpg");
    Tensor rgb_tensor_cpu(rgb_mat); // 调用此Tensor::Tensor(const cv::Mat& mat)构造函数

    RGB2YUVPipeline pipeline(std::move(rgb_tensor_cpu),  
                           rgb_mat.cols, rgb_mat.rows);
    
    Tensor yuv_tensor_cpu = pipeline.process(); // 移动赋值
    
    cv::imwrite("output.jpg", yuv_tensor_cpu.toMat(255.0f));
    return 0;
}