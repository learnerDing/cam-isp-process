//This is main3.cpp文件 用来测试Tensor.toMat方法的正确性
// test_tensor.cpp
#include "Tensor.h"
#include <opencv2/opencv.hpp>
#include<vector>
#include<iostream>
void test_yuv_to_rgb() {
    // 构造一个已知的YUV420P数据（示例：纯红色）
    const int width = 2, height = 2;
    const int y_size = width * height;
    const int uv_size = (width/2) * (height/2);
    
    // YUV值参考：RGB(255,0,0) -> YUV(76, 84, 255)
    std::vector<float> yuv_data = {
        /* Y */ 76/255.f, 76/255.f, 76/255.f, 76/255.f,
        /* U */ 84/255.f, 84/255.f,
        /* V */ 255/255.f, 255/255.f
    };
    
    // 创建Tensor
    Tensor test_tensor({1, 1, y_size + 2*uv_size}, DataType::FLOAT32, DeviceType::CPU);
    memcpy(test_tensor.data(), yuv_data.data(), test_tensor.bytes());
    test_tensor.print("test_tensor");
    test_tensor.width_ = width;
    test_tensor.height_ = height;
    
    // 转换为Mat
    cv::Mat rgb_mat = test_tensor.toMat(255.0f);
    
    // 验证输出是否为纯红色
    if (rgb_mat.at<cv::Vec3b>(0,0) == cv::Vec3b(0, 0, 255)) { // OpenCV使用BGR格式
        std::cout << "Test passed: Red color converted correctly!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    
    cv::imwrite("test_output.jpg", rgb_mat);
}

int main() {
    test_yuv_to_rgb();
    return 0;
}