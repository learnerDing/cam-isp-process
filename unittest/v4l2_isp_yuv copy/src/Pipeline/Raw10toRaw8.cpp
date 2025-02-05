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

int main() {
    // 定义图像的宽度和高度
    int width = 3264;
    int height = 2464;

    // 动态分配数组用于存储RAW10和RAW8格式的图像数据
    uint16_t* raw10 = new uint16_t[width * height];  // RAW10格式（16位）的数组
    uint8_t* raw8 = new uint8_t[width * height];     // RAW8格式（8位）的数组

    // 从文件中读取RAW10格式图像数据
    ifstream fin("1.raw", ios::binary);  // 打开RAW10图像的二进制文件
    fin.read((char*)raw10, width * height * sizeof(uint16_t));  // 读取文件中的RAW10数据
    fin.close();  // 读取完成后关闭文件

    // 调用函数将RAW10转换为RAW8
    raw10_to_raw8(raw10, raw8, width, height);

    // 将转换后的RAW8图像数据写入文件
    ofstream fout("raw8_image.raw", ios::binary);  // 创建或打开RAW8图像文件用于写入
    fout.write((char*)raw8, width * height * sizeof(uint8_t));  // 将RAW8数据写入文件
    fout.close();  // 写入完成后关闭文件

    // 释放动态分配的内存
    delete[] raw10;  // 释放存储RAW10数据的内存
    delete[] raw8;   // 释放存储RAW8数据的内存

    return 0;  // 程序正常结束
}