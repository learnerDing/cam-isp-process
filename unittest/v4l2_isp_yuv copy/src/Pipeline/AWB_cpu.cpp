#include <algorithm>
#include "../utils/Tensor.h"
#define Garyworld 0
#define Perfectreflex 1
// #define Imgwidth 3264
// #define Imgheight 2464
#define Imgwidth 200    //列数
#define Imgheight 100  //行数
#define ColorChannels 3
#ifdef DEBUG
#include <cstdio>
#endif
//tensordata格式：[1,RGBchannels=3,height,weight]

//分别求出三个通道的颜色均值,返回值为一个数组的地址，里面存放着三个RGB色彩均值
float* AWB_percolor_avg(TensorWrapper<uint8_t>* Imgdata) {
    #ifdef DEBUG
    printf("Entering AWB_percolor_avg\n");
#endif
    int height = Imgdata->shape[1];
    int width = Imgdata->shape[2];
    int num_pixels = height * width;
    float* color_avg = new float[3]{0.0f, 0.0f, 0.0f}; // 初始化为0

    // 累加每个通道的颜色值
    for (int i = 0; i < num_pixels; ++i) {
        color_avg[0] += Imgdata->getVal(i * 3);     // R通道
        color_avg[1] += Imgdata->getVal(i * 3 + 1); // G通道
        color_avg[2] += Imgdata->getVal(i * 3 + 2); // B通道
    }

    // 计算平均值
    color_avg[0] /= num_pixels;
    color_avg[1] /= num_pixels;
    color_avg[2] /= num_pixels;
    #ifdef DEBUG
    printf("Exiting AWB_percolor_avg with avg R: %f, G: %f, B: %f\n", color_avg[0], color_avg[1], color_avg[2]);
#endif
    return color_avg;
}



//一张图片红色蓝色通道按照增益放大,直接在Imgdata上面修改，减少拷贝
void AWB_RGchannelGain(TensorWrapper<uint8_t>* Imgdata, float r_gain, float b_gain) {
    #ifdef DEBUG
    printf("Entering AWB_RGchannelGain with r_gain: %f, b_gain: %f\n", r_gain, b_gain);
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

        // 修改 R 和 B 通道的值
        int new_r = static_cast<int>(r * r_gain);
        int new_b = static_cast<int>(b * b_gain);

        // 截断值使之处于 0-255 之间
        Imgdata->data[i * 3] = std::min(255, std::max(0, new_r));
        Imgdata->data[i * 3 + 1] = g; // G 通道不变
        Imgdata->data[i * 3 + 2] = std::min(255, std::max(0, new_b));

    }    
    #ifdef DEBUG
        printf("Exiting AWB_RGchannelGain\n");
    #endif
}

//总函数，对图像数据进行白平衡处理
void AWB_process(TensorWrapper<uint8_t>* Imgdata, bool GaryworldorPerfectref = Garyworld) {
    #ifdef DEBUG
    printf("Entering AWB_process\n");
    #endif
    float* color_avg = AWB_percolor_avg(Imgdata); //得到r, g, b通道的均值
    
    float r_gain = color_avg[1] / color_avg[0]; // G/R 得到增益
    float b_gain = color_avg[1] / color_avg[2]; // G/B 得到增益

    // 对 R 和 B 通道进行增益调整
    AWB_RGchannelGain(Imgdata, r_gain, b_gain);

    // 释放动态分配的内存
    delete[] color_avg;
    #ifdef DEBUG
    printf("Exiting AWB_process\n");
    #endif
}
int main()
{   uint8_t* ImgdataIn = (uint8_t*)malloc(sizeof(uint8_t)*ColorChannels*Imgheight*Imgwidth);
    uint8_t* ImgdataOut = (uint8_t*)malloc(sizeof(uint8_t)*ColorChannels*Imgheight*Imgwidth);//接收白平衡之后的图像数据
    for(int i = 0; i < ColorChannels*Imgheight*Imgwidth; i++){
        ImgdataIn[i] = i%50;
    }
    TensorWrapper<uint8_t>* AWB_Indata = new TensorWrapper<uint8_t>(CPU,UINT8,{1,ColorChannels,Imgheight,Imgwidth},ImgdataIn);
    
    for(int j=0;j<=200;j++)
    {
        printf("%d ",AWB_Indata->getVal(j));
    }
    AWB_process(AWB_Indata,Garyworld);
    printf("\n");
    TensorWrapper<uint8_t>* AWB_Outdata = new TensorWrapper<uint8_t>(AWB_Indata);
    for(int j=0;j<=200;j++)
    {
        printf("%d ",AWB_Outdata->getVal(j));
    }
}