#include <iostream>
#include <cstdlib> // for std::malloc, std::free
#include <algorithm> // for std::min, std::max>

using namespace std;
unsigned char* convertFloatToUnsignedChar(float* floatData, int channel,int width, int height) {
    // 计算图像的总像素数
    int totalPixels = width * height*channel;
    
    // 分配内存给转换后的unsigned char图像数据
    unsigned char* ucharData = (unsigned char*)std::malloc(totalPixels * sizeof(unsigned char));
    if (ucharData == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return nullptr; // 内存分配失败，返回nullptr
    }

    // 转换float数据到unsigned char
    for (int i = 0; i < totalPixels; i++) {
        // 假设float值在0到1之间进行归一化处理
        float value = floatData[i];
        if (value < 0.0f) {
            ucharData[i] = 0; // 小于0的值设为0
        } else if (value > 255.0f) {
            ucharData[i] = 255; // 大于1的值设为255
        } else {
            ucharData[i] = static_cast<unsigned char>(value); // 直接转换
        }
    }

    return ucharData; // 返回转换后的数据指针
}

float* convertUnsignedCharToFloat(unsigned char* ucharData, int channel,int width, int height) {
    // 计算图像的总像素数
    int totalPixels = width * height* channel;

    // 分配内存给转换后的 float 图像数据
    float* floatData = (float*)std::malloc(totalPixels * sizeof(float));
    if (floatData == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return nullptr; // 内存分配失败，返回 nullptr
    }

    // 转换 unsigned char 数据到 float 数据
    for (int i = 0; i < totalPixels; i++) {
        floatData[i] = static_cast<float>(ucharData[i]); // 直接转换
    }

    return floatData; // 返回转换后的数据指针
}
// 将 Tensor 数据转换为 OpenCV Mat 格式
cv::Mat Tensor2Mat(float* Tensordata, int channels = 3, int rows = 2464, int cols = 3264) {
    // 创建一个具有 HWC 格式的 Mat
    cv::Mat mat(rows, cols, CV_8UC3);
    
    // 转换 CHW 到 HWC
    for (int h = 0; h < rows; ++h) {
        for (int w = 0; w < cols; ++w) {
            for (int c = 0; c < channels; ++c) {
                // 归一化并将 float 数据转换为 unsigned char
                // if(w==3263){
                //     std::cout<<"c="<<c<<" w="<<w<<" h="<<h<<std::endl;}
                float tempdata = Tensordata[c * rows * cols + h * cols + w];
                float mindata = std::min(tempdata, 255.0f); // 将值缩放到 [0, 255]
                mat.at<cv::Vec3b>(h, w)[c]= static_cast<unsigned char>(mindata);
            }
        }
    }
    // 返回 Mat 的数据指针
    return mat;
}

// 将 OpenCV Mat 数据转换为 Tensor 格式
// float* Mat2Tensor(unsigned char* Matdata, int channels = 3, int rows = 2464, int cols = 3264) {
//     // 为 Tensor 分配内存
//     assert(Matdata!=NULL);
//     float* Tensordata = new float[channels * rows * cols];

//     // 转换 HWC 到 CHW
//     for (int h = 0; h < rows; ++h) {
//         for (int w = 0; w < cols; ++w) {
//             for (int c = 0; c < channels; ++c) {
//                 Tensordata[c * rows * cols + h * cols + w] = static_cast<float>(
//                     Matdata[h * cols * channels + w * channels + c]); 
//             }
//         }
//     }

//     return Tensordata;
// }
std::vector<float> Mat2Tensor(unsigned char* Matdata, int channels = 3, int rows = 2464, int cols = 3264) {
    assert(Matdata != NULL);
    assert(channels > 0 && rows > 0 && cols > 0);

    // 为 Tensor 分配内存
    std::vector<float> Tensordata(channels * rows * cols);

    // 转换 HWC 到 CHW
    for (int h = 0; h < rows; ++h) {
        for (int w = 0; w < cols; ++w) {
            for (int c = 0; c < channels; ++c) {
                Tensordata[c * rows * cols + h * cols + w] = static_cast<float>(
                    Matdata[h * cols * channels + w * channels + c]); 
            }
        }
    }

    return Tensordata; // 返回 std::vector，会自动管理内存
}
// //TensorWrapper存储的图像先转化排列然后调用mat.imshow显示
// // 从 Tensor 数据显示图像的函数
// void TensorPicShow(float* Tensordata, int channels = 3, int rows = 2464, int cols = 3264) {
//     // 将 Tensor 转换为 OpenCV Mat
//     unsigned char* matData = Tensor2Mat(Tensordata, channels, rows, cols);
    
//     // 使用数据创建 Mat 对象
//     cv::Mat image(rows, cols, CV_8UC3, matData);
    
//     cv::namedWindow("Resizable Window", cv::WINDOW_NORMAL);

//     // //设置窗口大小
//     // cv::resizeWindow("Resizable Window", 816, 616);
//     cv::imshow("Resizable Window", image);
//     cv::waitKey(0); // 等待键盘输入
// }