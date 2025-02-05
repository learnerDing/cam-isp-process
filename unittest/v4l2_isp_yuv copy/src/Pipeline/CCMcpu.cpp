#include <algorithm>
#include "../utils/Tensor.h"
#define Imgwidth 200    //列数
#define Imgheight 100   //行数
#define ColorChannels 3
#ifdef DEBUG
#include <cstdio>
#endif

// 使用色彩校正矩阵对图像数据进行色彩变换
void CCM_apply(TensorWrapper<uint8_t>* Imgdata, TensorWrapper<float>* ccm) {
    #ifdef DEBUG
    printf("Entering CCM_apply\n");
    #endif
    int height = Imgdata->shape[1];
    int width = Imgdata->shape[2];
    int num_pixels = height * width;

    // 逐像素处理
    for (int i = 0; i < num_pixels; ++i) {
        // 获取原 RGB 值
        float r = Imgdata->getVal(i * 3) / 255.0f; // 转换为浮点值以避免溢出
        float g = Imgdata->getVal(i * 3 + 1) / 255.0f;
        float b = Imgdata->getVal(i * 3 + 2) / 255.0f;

        // 使用给定的色彩校正矩阵进行计算
        float new_r = ccm->getVal(0) * r + ccm->getVal(1) * g + ccm->getVal(2) * b; // R'
        float new_g = ccm->getVal(3) * r + ccm->getVal(4) * g + ccm->getVal(5) * b; // G'
        float new_b = ccm->getVal(6) * r + ccm->getVal(7) * g + ccm->getVal(8) * b; // B'

        // 将结果转回 0-255 范围，并更新图像数据
        Imgdata->data[i * 3] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, new_r * 255.0f)));
        Imgdata->data[i * 3 + 1] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, new_g * 255.0f)));
        Imgdata->data[i * 3 + 2] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, new_b * 255.0f)));
    }    
    #ifdef DEBUG
    printf("Exiting CCM_apply\n");
    #endif
}

// 总函数，对图像数据进行颜色校正处理
void CCM_process(TensorWrapper<uint8_t>* Imgdata, TensorWrapper<float>* ccm) {
    #ifdef DEBUG
    printf("Entering CCM_process\n");
    #endif
    
    // 进行色彩校正
    CCM_apply(Imgdata, ccm);

    #ifdef DEBUG
    printf("Exiting CCM_process\n");
    #endif
}

int main() {
    uint8_t* ImgdataIn = (uint8_t*)malloc(sizeof(uint8_t) * ColorChannels * Imgheight * Imgwidth);
    uint8_t* ImgdataOut = (uint8_t*)malloc(sizeof(uint8_t) * ColorChannels * Imgheight * Imgwidth); // 接收颜色校正之后的图像数据

    // 初始化输入图像数据
    for (int i = 0; i < ColorChannels * Imgheight * Imgwidth; i++) {
        ImgdataIn[i] = i % 50; // 示例数据
    }
    
    // 创建输入张量
    TensorWrapper<uint8_t>* CCM_Indata = new TensorWrapper<uint8_t>(CPU, UINT8, {1, ColorChannels, Imgheight, Imgwidth}, ImgdataIn);

    // 打印输入数据
    for (int j = 0; j <= 200; j++) {
        printf("%d ", CCM_Indata->getVal(j));
    }
    
    // 创建色彩校正矩阵
    float ccm_data[9] = {
        1.0f, 0.0f, 0.0f, // 第一列
        0.0f, 1.0f, 0.0f, // 第二列
        0.0f, 0.0f, 1.0f  // 第三列
        // 请根据需要替换为自定义的CCM数值
    };

    TensorWrapper<float>* ccm = new TensorWrapper<float>(CPU, FP32, {1, 3, 3}, ccm_data); // 3x3 的色彩校正矩阵

    // 进行颜色校正处理
    CCM_process(CCM_Indata, ccm);

    printf("\n");
    
    // 打印处理后的数据
    for (int j = 0; j <= 200; j++) {
        printf("%d ", CCM_Indata->getVal(j)); // 输出处理后的图像数据
    }

    // 清理内存
    delete CCM_Indata;
    delete ccm;
    free(ImgdataIn);
    free(ImgdataOut);

    return 0;
}