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

// // 测试示例程序
// int main() {
//     int width = 2;
//     int height = 2;
//     unsigned char ucharData[4] = {0, 255, 128, 64}; // 测试数据

//     float* floatData = convertUnsignedCharToFloat(ucharData, width, height);
    
//     if (floatData) {
//         for (int i = 0; i < width * height; ++i) {
//             std::cout << floatData[i] << " "; // 输出转化后的结果
//         }
//         std::cout << std::endl;

//         // 释放分配的内存
//         std::free(floatData);
//     }

//     return 0;
// }
#include <iostream>
using namespace std;

template <typename T>
void PrintValues(T* ptr, int begin, int end) {
    // 检查指针是否为 nullptr
    if (ptr == nullptr) {
        cout << "Pointer is null." << endl;
        return;
    }

    // 检查索引范围的有效性
    if (begin < 0 || end < begin) {
        cout << "Invalid range." << endl;
        return;
    }

    // 打印指定范围内的值
    for (int i = begin; i <= end; i++) {
        cout << "Value at index " << i << ": " << static_cast<typename std::conditional<std::is_same<T, unsigned char>::value, int, T>::type>(ptr[i]) << endl;
    }
}
template void PrintValues(float* ptr, int begin, int end) ;
template void PrintValues(unsigned char* ptr, int begin, int end) ;