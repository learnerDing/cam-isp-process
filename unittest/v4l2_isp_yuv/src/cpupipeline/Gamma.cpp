#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../utils/Tensor.h"
#define Imgwidth 200    // 列数
#define Imgheight 100   // 行数
#define ColorChannels 3
#define GAMMA 2.2f      // Gamma 值

#ifdef DEBUG
#include <cstdio>
#endif

// 使用 gamma 矫正查找表对图像数据进行处理
void gamma_correction(TensorWrapper<uint8_t>* Imgdata, const std::vector<uint8_t>& lut) {//查找表（LUT）
    #ifdef DEBUG
    printf("Entering gamma_correction\n");
    #endif

    int height = Imgdata->shape[1];
    int width = Imgdata->shape[2];
    int num_pixels = height * width;

    // 逐像素处理
    for (int i = 0; i < num_pixels; ++i) {
        // 获取原 RGB 值
        uint8_t r = Imgdata->getVal(i * 3);
        uint8_t g = Imgdata->getVal(i * 3 + 1);
        uint8_t b = Imgdata->getVal(i * 3 + 2);

        // 使用 LUT 进行 gamma 矫正
        Imgdata->data[i * 3] = lut[r];
        Imgdata->data[i * 3 + 1] = lut[g];
        Imgdata->data[i * 3 + 2] = lut[b];
    }

    #ifdef DEBUG
    printf("Exiting gamma_correction\n");
    #endif
}

// 总函数，对图像数据进行 gamma 矫正处理
void gamma_process(TensorWrapper<uint8_t>* Imgdata, const std::vector<uint8_t>& lut) {
    #ifdef DEBUG
    printf("Entering gamma_process\n");
    #endif
    
    // 进行 gamma 矫正
    gamma_correction(Imgdata, lut);

    #ifdef DEBUG
    printf("Exiting gamma_process\n");
    #endif
}

// 生成 gamma 校正查找表
std::vector<uint8_t> generate_gamma_lut(float gamma) {
    std::vector<uint8_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        lut[i] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, std::pow(i / 255.0f, gamma) * 255.0f)));
    }
    return lut;
}

int main() {
    uint8_t* ImgdataIn = (uint8_t*)malloc(sizeof(uint8_t) * ColorChannels * Imgheight * Imgwidth);
    
    // 初始化输入图像数据
    for (int i = 0; i < ColorChannels * Imgheight * Imgwidth; i++) {
        ImgdataIn[i] = i % 256; // 示例数据
    }
    
    // 创建输入张量
    TensorWrapper<uint8_t>* gamma_Indata = new TensorWrapper<uint8_t>(CPU, UINT8, {1, ColorChannels, Imgheight, Imgwidth}, ImgdataIn);
    
    // 生成 gamma 校正查找表
    std::vector<uint8_t> gamma_lut = generate_gamma_lut(GAMMA);

    // 进行 gamma 矫正处理
    gamma_process(gamma_Indata, gamma_lut);

    printf("\n");
    
    // 打印处理后的数据
    for (int j = 0; j <= 200; j++) {
        printf("%d ", gamma_Indata->getVal(j)); // 输出处理后的图像数据
    }

    // 清理内存
    delete gamma_Indata;
    free(ImgdataIn);

    return 0;
}