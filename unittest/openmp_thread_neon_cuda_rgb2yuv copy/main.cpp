// main.cpp
#include <opencv2/opencv.hpp>
#include "rgb2yuvpipeline.h"

int main() {
    // 读取输入图像
    cv::Mat rgb_mat = cv::imread("input.jpg");
    if(rgb_mat.empty()) return -1;

    // 创建cpuTensor并转换到float32
    Tensor rgb_tensor(rgb_mat, 1.0f/255.0f); // 自动归一化到[0,1]
    Tensor yuv_tensor;
    // 创建处理管线
    const int width = rgb_mat.cols;
    const int height = rgb_mat.rows;
    RGB2YUVPipeline pipeline(rgb_tensor,yuv_tensor,width, height);

    // 处理图像
    // Tensor yuv_tensor;
     yuv_tensor = pipeline.process();
    // 转换为Mat并保存结果
    cv::Mat yuv_mat = yuv_tensor.toMat(255.0f); // 反归一化
    cv::imwrite("output.jpg", yuv_mat);

    return 0;
}