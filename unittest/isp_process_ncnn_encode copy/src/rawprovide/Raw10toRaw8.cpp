#include <iostream>
#include <fstream>
#include <stdint.h>

using namespace std;

// 将RAW10格式转换为RAW8格式的函数
void raw10_to_raw8(uint16_t* raw10, uint8_t* raw8, int width, int height) {
    // 遍历所有像素，进行转换
    for (int i = 0; i < width * height; i++) {
        // RAW10是10位数据，而RAW8是8位数据，这里通过右移2位将RAW10转换为RAW8
        raw8[i] = raw10[i] >> 2;  // 舍弃RAW10的高两位
    }
}

// 截取图像的函数
void crop_image(uint16_t* src, uint16_t* dst, int src_width, int src_height, int dst_width, int dst_height, int start_x, int start_y) {
    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            dst[y * dst_width + x] = src[(start_y + y) * src_width + (start_x + x)];
        }
    }
}

int main() {
    // 定义原始图像的宽度和高度
    int src_width = 3264;
    int src_height = 2464;

    // 定义目标图像的宽度和高度
    int dst_width = 1920;
    int dst_height = 1080;

    // 定义截取的起始位置（假设从中心截取）
    int start_x = (src_width - dst_width) / 2;
    int start_y = (src_height - dst_height) / 2;

    // 处理五张照片
    for (int i = 1; i <= 5; i++) {
        // 动态分配数组用于存储RAW10和RAW8格式的图像数据
        uint16_t* raw10 = new uint16_t[src_width * src_height];  // RAW10格式（16位）的数组
        uint16_t* cropped_raw10 = new uint16_t[dst_width * dst_height];  // 截取后的RAW10数据
        uint8_t* raw8 = new uint8_t[dst_width * dst_height];     // RAW8格式（8位）的数组

        // 从文件中读取RAW10格式图像数据
        string input_filename = "3264_2464_10_" + to_string(i) + ".raw";
        ifstream fin(input_filename, ios::binary);  // 打开RAW10图像的二进制文件
        fin.read((char*)raw10, src_width * src_height * sizeof(uint16_t));  // 读取文件中的RAW10数据
        fin.close();  // 读取完成后关闭文件

        // 截取图像
        crop_image(raw10, cropped_raw10, src_width, src_height, dst_width, dst_height, start_x, start_y);

        // 调用函数将RAW10转换为RAW8
        raw10_to_raw8(cropped_raw10, raw8, dst_width, dst_height);

        // 将转换后的RAW8图像数据写入文件
        string output_filename = "1920_1080_8_" + to_string(i) + ".raw";
        ofstream fout(output_filename, ios::binary);  // 创建或打开RAW8图像文件用于写入
        fout.write((char*)raw8, dst_width * dst_height * sizeof(uint8_t));  // 将RAW8数据写入文件
        fout.close();  // 写入完成后关闭文件

        // 释放动态分配的内存
        delete[] raw10;
        delete[] cropped_raw10;
        delete[] raw8;
    }

    return 0;  // 程序正常结束
}