// This is main.cpp
#include <opencv2/opencv.hpp>
#include "rgb2yuvpipeline.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <fstream>
// main.cpp
int main() {
    // Tensor::enable_debug(true);
    int ret;
    cv::Mat rgb_mat = cv::imread("../input.jpg");
    Tensor rgb_tensor_cpu(rgb_mat); // 调用此Tensor::Tensor(const cv::Mat& mat)构造函数
    //此时rgb_tensor_cpu已经是chw排列
    //构建RGB2YUVPipeline
    RGB2YUVPipeline pipeline(std::move(rgb_tensor_cpu),  
                           rgb_mat.cols, rgb_mat.rows);
     // 移动赋值 , yuv_tensor_cpu shape为[1,1,1.5*width*height]的yuv420p的shape
    Tensor yuv_tensor_cpu = pipeline.process();
    
    // 新增：将YUV420P原始数据保存为文件
    const int y_size = rgb_mat.rows * rgb_mat.cols;
    const int uv_size = (rgb_mat.rows/2) * (rgb_mat.cols/2);
    const float* yuv_data = static_cast<const float*>(yuv_tensor_cpu.data());
    
    // 转换为字节并保存
    std::vector<uchar> yuv_bytes(y_size + 2*uv_size);
    for (int i = 0; i < y_size + 2*uv_size; ++i) {
        yuv_bytes[i] = static_cast<uchar>(yuv_data[i] * 255.0f);
    }
    
    std::ofstream fout("output.yuv", std::ios::binary);
    fout.write(reinterpret_cast<const char*>(yuv_bytes.data()), yuv_bytes.size());
    fout.close();
    
    return 0;
}
