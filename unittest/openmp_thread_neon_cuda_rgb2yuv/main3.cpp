//This is main3.cpp文件 用来测试
#include <opencv2/opencv.hpp>
#include "bgr2yuvpipeline.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <fstream>

int main() {
    // 创建4x4蓝色测试图像（BGR格式）
    const int SIZE = 4;
    cv::Mat blue_mat(SIZE, SIZE, CV_8UC3, cv::Scalar(255, 0, 0)); // BGR蓝色

    // 转换为Tensor（自动执行CHW转换和归一化）
    Tensor bgr_tensor_cpu(blue_mat);
    
    // 构建处理流水线
    BGR2YUVPipeline pipeline(std::move(bgr_tensor_cpu), SIZE, SIZE);
    Tensor yuv_tensor_cpu = pipeline.process();

    // 验证输出数据格式
    const auto& yuv_shape = yuv_tensor_cpu.shape();
    const int expected_yuv_size = SIZE*SIZE + 2*(SIZE/2)*(SIZE/2);
    assert(yuv_shape.size() == 3 && 
          yuv_shape[0] == 1 && 
          yuv_shape[1] == 1 && 
          yuv_shape[2] == expected_yuv_size);

    // 打印调试信息
    std::cout << "=== Input BGR Tensor ===" << std::endl;
    blue_mat.convertTo(blue_mat, CV_32FC3, 1.0/255); // 转换为float用于验证
    std::cout << "Original BGR data (HWC format):\n" << blue_mat << "\n";
    bgr_tensor_cpu.print("BGR Tensor", 16); // 打印前16个元素

    std::cout << "\n=== Output YUV420P Tensor ===" << std::endl;
    yuv_tensor_cpu.print("YUV420P Tensor", 16);

    // 保存原始YUV数据（可选）
    const float* yuv_data = static_cast<const float*>(yuv_tensor_cpu.data());
    std::vector<uchar> yuv_bytes(expected_yuv_size);
    for (int i = 0; i < expected_yuv_size; ++i) {
        yuv_bytes[i] = static_cast<uchar>(yuv_data[i] * 255.0f);
    }
    std::ofstream("blue_4x4.yuv", std::ios::binary)
        .write(reinterpret_cast<const char*>(yuv_bytes.data()), yuv_bytes.size());

    return 0;
}