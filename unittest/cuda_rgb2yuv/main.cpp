// This is main.cpp
#include <opencv2/opencv.hpp>
#include "bgr2yuvpipeline.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <fstream>
// main.cpp
int main() {
    // Tensor::enable_debug(true);
    int ret;
    cv::Mat bgr_mat = cv::imread("../input.jpg");
    Tensor bgr_tensor_cpu(bgr_mat); // 调用此Tensor::Tensor(const cv::Mat& mat)构造函数
    //此时bgr_tensor_cpu已经是chw排列
    //构建BGR2YUVPipeline
    BGR2YUVPipeline pipeline(std::move(bgr_tensor_cpu),  
                           bgr_mat.cols, bgr_mat.rows);
     // 移动赋值 , yuv_tensor_cpu shape为[1,1,1.5*width*height]的yuv420p的shape
    Tensor yuv_tensor_cpu = pipeline.process();
    
    // 新增：将YUV420P原始数据保存为文件
    const int y_size = bgr_mat.rows * bgr_mat.cols;
    const int uv_size = (bgr_mat.rows/2) * (bgr_mat.cols/2);
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
