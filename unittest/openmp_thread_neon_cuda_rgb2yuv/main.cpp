// This is main.cpp
#include <opencv2/opencv.hpp>
#include "rgb2yuvpipeline.h"

// main.cpp
int main() {
    cv::Mat rgb_mat = cv::imread("input.jpg");
    Tensor rgb_tensor_cpu(rgb_mat); // 自动转换
    Tensor yuv_tensor_cpu;

    RGB2YUVPipeline pipeline(rgb_tensor_cpu, yuv_tensor_cpu, 
                           rgb_mat.cols, rgb_mat.rows);
    
    yuv_tensor_cpu = pipeline.process(); // 移动赋值
    
    cv::imwrite("output.jpg", yuv_tensor_cpu.toMat(255.0f));
    return 0;
}