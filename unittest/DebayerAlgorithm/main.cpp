//This is main.cpp
//此文件使用opencv的debayer和自己写的debayer文件做对比
#include <iostream>
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "RawScaler.h"


using namespace std;
using namespace cv;
#define channel 3
#define col 3264
#define row 2464
#define MyBayer2RGB


//截断函数
void opencvMatPrint(cv::Mat image)
{
    cout << image<<"cols="<<image.cols<< endl;         //相片的列数，一共有多少列，对应width
    cout << image<<"rows="<<image.rows << endl;         //相片的行数，一共有多少行，对应height
    cout << image<<"dims="<<image.dims << endl;         //相片的维度，这里都是二维
     
    cout << image<<"type="<<image.type() << endl;       //16,其实是CV_8U3，即8位无符号三通道
    cout << image<<"pixelnum="<<image.total() << endl;      //156500  总共有多少个元素，即rows*cols
    cout << image<<"channels="<<image.channels() << endl;   //相片的通道数，可能是1、3、4
        
}

float clamp(float value, float min, float max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

#define mBayerType   BGGR; //GRBG,BGGR,RGGB,GBRG
cv::Mat CVReadraw()
{
    int rows = 2464;  // 图像的行数
    int cols = 3264;  // 图像的列数
    int channels = 1; // 图像的通道数，灰度图为1

    // 从raw文件中读取数据
    std::string path = "raw8_image.raw";
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
cv::Mat ConvertBayer2RGB(cv::Mat bayer)
{
	cv::Mat RGB;// (bayer.rows, bayer.cols, CV_8UC3);
	cv::cvtColor(bayer, RGB, cv::COLOR_BayerRG2RGB);
    return RGB;
}

int main()
{
    // 1. 读取raw格式图像
    cv::Mat rawfileImage = CVReadraw();
    if (rawfileImage.empty()) {
        return -1; // 如果读取失败，退出
    }
#ifdef MyBayer2RGB
    cout <<"MyBayer2RGB"<<endl;
    cv::Mat rawImage = rawfileImage.clone();//深拷贝一张后面就可以直接覆盖修改了
    unsigned char* Matdata = rawImage.data;

    cv::Mat bgrmat(2464,3264,CV_8UC3,Raw2RGBEntry(Matdata,2464,3264));//根据指针创建Mat,格式为bgr888
    // PrintValues(RGBImagedata,0,100);
    // PrintValues(RGBImagedata,4021148,4021248);
    // PrintValues(RGBImagedata,2010524,2010624);

#else
    // 2. 完成Bayer到RGB的插值
    cv::Mat rgbImage = ConvertBayer2RGB(rawImage);
    opencvMatPrint(rgbImage);
#endif
    cv::namedWindow("Original Raw Image", cv::WINDOW_NORMAL);
    cv::imshow("Original Raw Image", rawImage);

    cv::namedWindow("BGR Image", cv::WINDOW_NORMAL);
    cv::imshow("BGR Image", bgrmat);
    cv::waitKey(0); // 等待按键
    return 0;
}