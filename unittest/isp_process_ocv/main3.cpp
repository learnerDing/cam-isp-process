//this is main.cpp
#include <iostream>
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "include/debayerpipeline.h"
#include <chrono> // 添加计时库
using namespace std;
using namespace cv;

#define width 1920
#define height 1080
#define MyBayer2RGB

// 截断函数
float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// 打印Mat数据结构的某行某个通道的值
void matprint(uchar* rowptr, int start_col, int end_col, int channel = 0) {
    if (!rowptr) {
        std::cerr << "Error: Row pointer is null!" << std::endl;
        return;
    }
    if (start_col > end_col) {
        std::cerr << "Error: start_col > end_col!" << std::endl;
        return;
    }
    if (channel < 0 || channel >= 3) {
        std::cerr << "Error: Invalid channel! Must be 0, 1, or 2." << std::endl;
        return;
    }
    std::cout << "Print Mat (Channel " << channel << "):" << std::endl;
    for (int col = start_col; col <= end_col; ++col) {
        int position = col * 3 + channel; // 根据通道偏移
        std::cout << static_cast<int>(rowptr[position]) << " ";
    }
    std::cout << std::endl;
}

#define mBayerType   RGGB; // GRBG,BGGR,RGGB,GBRG

// 自己生成虚假raw8格式数据, 添加绘制彩色圆圈的功能
cv::Mat CVFakeraw() {
    cv::Mat fakeRawImage = cv::Mat::zeros(height, width, CV_8UC1);
    cv::Point center(width / 2, height / 2);
    int radius = std::min(width, height) / 4;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (pow(x - center.x, 2) + pow(y - center.y, 2) <= pow(radius, 2)) {
                // 根据Bayer RGGB模式设置像素值
                if (y % 2 == 0) { // 偶数行
                    if (x % 2 == 0) { // R通道
                        fakeRawImage.at<uchar>(y, x) = 100; // BGR中的R通道
                    } else { // G通道
                        fakeRawImage.at<uchar>(y, x) = 150; // BGR中的G通道
                    }
                } else { // 奇数行
                    if (x % 2 == 0) { // G通道
                        fakeRawImage.at<uchar>(y, x) = 150; // BGR中的G通道
                    } else { // B通道
                        fakeRawImage.at<uchar>(y, x) = 200; // BGR中的B通道
                    }
                }
            }
        }
    }
    return fakeRawImage;
}

cv::Mat CVReadraw() {
    int rows = 1080;  // 图像的行数
    int cols = 1920;  // 图像的列数
    int channels = 1; // 图像的通道数，灰度图为1
    // 从raw文件中读取数据
    std::string path = "../src/rawpic/1920_1080_8_1.raw";
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << path << std::endl;
        return cv::Mat(); // 返回一个空的 cv::Mat 对象
    }
    // 创建一个矩阵以存储图像数据
    cv::Mat img(rows, cols, CV_8UC1); // CV_8UC1: 8位无符号单通道
    // 读取数据
    file.read(reinterpret_cast<char*>(img.data), rows * cols);
    file.close();
    std::cout << "Read Raw ok" << std::endl;
    return img; // 返回读取到的图像
}

cv::Mat ConvertBayer2BGR(cv::Mat bayer) {
    cv::Mat BGR;
    cv::cvtColor(bayer, BGR, cv::COLOR_BayerRG2BGR); // 使用 BGR 通道顺序
    return BGR;
}

int main() {
    // 1. 读取raw格式图像
    cv::Mat rawfileImage = CVReadraw(); // 注释掉读取真实 raw 文件的代码
    // cv::Mat rawfileImage = CVFakeraw();  // 使用自己生成的 fake raw 图像
    if (rawfileImage.empty()) {
        return -1; // 如果读取/生成失败，退出
    }
    cout << "MyBayer2BGR" << endl;
    cv::Mat rawImage = rawfileImage.clone(); // 深拷贝一张后面就可以直接覆盖修改了

#ifdef MyBayer2RGB
    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    Tensor bayer_tensor_cpu(rawImage); // 从rawImage创建Tensor，深拷贝
    // 创建 DebayerPipeline 对象
    DebayerPipeline pipeline(std::move(bayer_tensor_cpu), width, height);
    // 执行 Debayer 操作并获取结果
    Tensor bgr_tensor_cpu = pipeline.process();
    std::cout << "debayer over!" << std::endl;
    cv::Mat bgr_mat = bgr_tensor_cpu.toMat(255.0);

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "MyBayer2RGB Processing Time: " << elapsed.count() << " ms" << std::endl;

    matprint(bgr_mat.ptr<uchar>(540), 940, 1000, 0);
    matprint(bgr_mat.ptr<uchar>(540), 940, 1000, 1);
    matprint(bgr_mat.ptr<uchar>(540), 940, 1000, 2);
#else
    // 计时开始
    auto start = std::chrono::high_resolution_clock::now();

    // 2. 完成Bayer到RGB的插值
    cv::Mat bgr_mat = ConvertBayer2BGR(rawImage);

    // 计时结束
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "OpenCV Debayer Processing Time: " << elapsed.count() << " ms" << std::endl;

    matprint(bgr_mat.ptr<uchar>(540), 940, 1000, 0);
    matprint(bgr_mat.ptr<uchar>(540), 940, 1000, 1);
    matprint(bgr_mat.ptr<uchar>(540), 940, 1000, 2);
#endif

    cv::namedWindow("Original Raw Image", cv::WINDOW_NORMAL);
    cv::imshow("Original Raw Image", rawfileImage); // 显示 fake raw 图像
    cv::namedWindow("BGR Image", cv::WINDOW_NORMAL);
    cv::imshow("BGR Image", bgr_mat);
    cv::waitKey(0); // 等待按键
    return 0;
}