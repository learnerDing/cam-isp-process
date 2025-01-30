//This is main3.cpp文件 用来测试
#include <opencv2/opencv.hpp>
#include "bgr2yuvpipeline.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <fstream>

int main() {
    // 创建1024x1024蓝色测试图像（BGR格式）
    const int SIZE = 4;
    cv::Mat blue_mat(SIZE, SIZE, CV_8UC3, cv::Scalar(255, 0, 0)); // BGR蓝色

    // 转换为Tensor（自动执行CHW转换和归一化）
    Tensor bgr_tensor_cpu(blue_mat);
    bgr_tensor_cpu.print(0,0,2,"bgr_tensor_cpu");
    // 构建处理流水线
    BGR2YUVPipeline pipeline(std::move(bgr_tensor_cpu), SIZE, SIZE);
    Tensor yuv_tensor_cpu = pipeline.process();
    yuv_tensor_cpu.print(0,0,24,"yuv_tensor_cpu");
    /*对于蓝色来说BGR(255,0,0)对于Y=29,归一化=0.1137,U = -56（归一化：约0.0）;V  ≈ 105（归一化：约0.412）*/
    // 验证输出数据格式
    const auto& yuv_shape = yuv_tensor_cpu.shape();
    const int expected_yuv_size = SIZE*SIZE + 2*(SIZE/2)*(SIZE/2);
    assert(yuv_shape.size() == 3 && 
          yuv_shape[0] == 1 && 
          yuv_shape[1] == 1 && 
          yuv_shape[2] == expected_yuv_size);

    // 保存原始YUV数据（可选）
    const float* yuv_data = static_cast<const float*>(yuv_tensor_cpu.data());
    std::vector<uchar> yuv_bytes(expected_yuv_size);
    for (int i = 0; i < expected_yuv_size; ++i) {
        yuv_bytes[i] = static_cast<uchar>(yuv_data[i] * 255.0f);
    }
    std::ofstream("blue.yuv", std::ios::binary)
        .write(reinterpret_cast<const char*>(yuv_bytes.data()), yuv_bytes.size());

    // 新增部分：YUV转JPG
    // 读取保存的YUV文件
    std::ifstream yuv_file("blue.yuv", std::ios::binary | std::ios::ate);
    std::streamsize file_size = yuv_file.tellg();
    yuv_file.seekg(0, std::ios::beg);

    std::vector<uchar> yuv_data1(file_size);
    if (!yuv_file.read(reinterpret_cast<char*>(yuv_data1.data()), file_size)) {
        std::cerr << "Failed to read YUV file!" << std::endl;
        return -1;
    }

    const int width = SIZE;
    const int height = SIZE;

    // 创建OpenCV的YUV420P（I420）格式矩阵
    // 注意：I420格式需要高度为1.5倍原始高度（Y分量占满高度，UV各占1/4高度）
    cv::Mat yuv_i420(height * 3/2, width, CV_8UC1, yuv_data1.data());

    // 转换为BGR格式
    cv::Mat bgr_mat;
    cv::cvtColor(yuv_i420, bgr_mat, cv::COLOR_YUV2BGR_I420);

    // 保存为JPG
    if (!cv::imwrite("output.jpg", bgr_mat)) {
        std::cerr << "Failed to write JPG file!" << std::endl;
        return -1;
    }

    return 0;
}