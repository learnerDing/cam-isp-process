//此文件使用Opencv去完成图像isp过程
#include <iostream>
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include "Tensor.h"
#include "opencvauxiliary.h"
#include "RawScaler.h"
#include "AWBpipeline.h"
#include "CCMpipeline.h"
#include "Gammapipeline.h"

using namespace std;
using namespace cv;
#define channel 3
#define col 3264
#define row 2464
#define MyBayer2RGB
#define MyAWB
#define MyCCM
#define MyGamma

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
//自动白平衡灰度世界算法
void GrayWorldAlgorithm(const cv::Mat& src,cv::Mat& dst)
{
    assert(3==src.channels());

    //求BGR分量均值
    auto mean = cv::mean(src);

    //需要调整的BGR分量的增益
    float gain_B(0),gain_G(0),gain_R(0);//等效于gain_B=0
    float K = (mean[0]+mean[1]+mean[2])/3.0f;
    gain_B = K/mean[0];
    gain_G = K/mean[1];
    gain_R = K/mean[2];

    std::vector<cv::Mat> channels;
    cv::split(src,channels);

    //调整三个通道各自的值
    channels[0] = channels[0]*gain_B;
    channels[1] = channels[1]*gain_G;
    channels[2] = channels[2]*gain_R;

    //通道合并
    cv::merge(channels,dst);
}
cv::Mat ColorMatrixCorrect(cv::Mat imgin)
{
    int rows = imgin.rows;
    int cols = imgin.cols;
    cout << "image1 row0  col 0,2 = "<< endl << " "  << imgin.rowRange(0, 1).colRange(0,2) << endl << endl;
	cout << "image1 row0  col 1200,1202 = "<< endl << " "  << imgin.rowRange(0, 1).colRange(1200,1202) << endl << endl;
    float ccmmat[3][3] = {{2.34403f, 0.00823594f, -0.0795542f},
    {-1.18042f, 1.44385f, 0.0806464f},
    {-0.296824f, -0.556513f, 0.909063f}};
    //特殊CCM矩阵使得RGB保持不变，用于调试
    // float ccmmat[3][3] = {{1.0f, 0.0f, 0.0f},
    // {0.0f, 1.0f, 0.0f},
    // {0.0f, 0.0f, 1.0f}};
    cv::Mat CCM(3,3,CV_32FC1,ccmmat);// 颜色矫正矩阵3*3
    cout << "CCM"<< endl << " "  << CCM.rowRange(0, 3) << endl << endl;
    cv::Mat imgout(rows,cols, CV_8UC3); // 8位无符号三通道
    // 遍历每个像素
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            // 获取输入图像的像素值
            cv::Vec3b pixel = imgin.at<cv::Vec3b>(r, c);
            // float b = pixel[0] / 255.0f; // 将值归一化到[0, 1]
            // float g = pixel[1] / 255.0f;
            // float r = pixel[2] / 255.0f;
            float blue = pixel[0];//不需要除以 255.0f；这里已经是 [0, 255] 的范围
            float green = pixel[1];
            float red = pixel[2];
            // 应用颜色校正矩阵
           float new_r = CCM.at<float>(0, 0) * red + CCM.at<float>(1, 0) * green + CCM.at<float>(2, 0) * blue;
            float new_g = CCM.at<float>(0, 1) * red + CCM.at<float>(1, 1) * green + CCM.at<float>(2, 1) * blue;
            float new_b = CCM.at<float>(0, 2) * red + CCM.at<float>(1, 2) * green + CCM.at<float>(2, 2) * blue;
            // 确保输出值在[0, 255]范围内
            // new_r = clamp(new_r * 255.0f, 0.0f, 255.0f);
            // new_g = clamp(new_g * 255.0f, 0.0f, 255.0f);
            // new_b = clamp(new_b * 255.0f, 0.0f, 255.0f);
            new_r = clamp(new_r , 0.0f, 255.0f);//前面没有归一化
            new_g = clamp(new_g , 0.0f, 255.0f);
            new_b = clamp(new_b , 0.0f, 255.0f);
            // 将新值赋值给输出图像
            imgout.at<cv::Vec3b>(r, c) = cv::Vec3b(static_cast<uchar>(new_b), static_cast<uchar>(new_g), static_cast<uchar>(new_r));
        }
    }
    cout << "image2 row0  col 0,2 = "<< endl << " "  << imgout.rowRange(0, 1).colRange(0,2) << endl << endl;
	cout << "image2 row0  col 1200,1202 = "<< endl << " "  << imgout.rowRange(0, 1).colRange(1200,1202) << endl << endl;
    return imgout;
}
// Gamma变换函数实现
Mat gammaTransform(Mat& src, float kFactor=0.3f)
{
    // 建立查表文件LUT
    unsigned char LUT[256];
    for( int i = 0; i < 256; i++ )
    {
        // Gamma变换表达式
        LUT[i] = saturate_cast<uchar>(pow((float)(i / 255.0 ), kFactor ) * 255.0f);

    }
    //PrintValues(LUT,0,255);
    Mat resultImage = src.clone();
    // 输入通道为单通道时 直接进行变换
    if(src.channels() == 1)
    {
        MatIterator_<uchar> iterator = resultImage.begin<uchar>();
        MatIterator_<uchar> iteratorEnd = resultImage.end<uchar>();
        for( ; iterator != iteratorEnd; iterator++)
        {
            *iterator = LUT[(*iterator)];
        }     
    }
    else
    {
        // 输入通道为三通道时 需对每个通道分别进行变换
        MatIterator_<Vec3b> iterator = resultImage.begin<Vec3b>();
        MatIterator_<Vec3b> iteratorEnd = resultImage.end<Vec3b>();
        // 通过查找表进行转换
        for( ; iterator != iteratorEnd; iterator++)
        {
            (*iterator)[0] = LUT[(*iterator)[0]];
            (*iterator)[1] = LUT[(*iterator)[1]];
            (*iterator)[2] = LUT[(*iterator)[2]];
        }
    }
    return resultImage;
}

//isp使用opencv处理的主函数，调用上面的函数进行处理
int main()
{
    // 1. 读取raw格式图像
    cv::Mat rawImage = CVReadraw();
    if (rawImage.empty()) {
        return -1; // 如果读取失败，退出
    }
#ifdef MyBayer2RGB
    cout <<"MyBayer2RGB"<<endl;
    unsigned char* Matdata = rawImage.data;
    float* RGBImagedata = static_cast<float*>(malloc(sizeof(float)*channel*col*row)) ;
    // TensorWrapper<float>* RawTensor = new TensorWrapper<float>(Device::CPU,
    //                                                                 FP32,
    //                                                                 {1,2464,3264},
    //                                                                 RawImagedata);
    // TensorWrapper<float>* RGBTensor = new TensorWrapper<float>(Device::CPU,
    //                                                                 FP32,
    //                                                                 {3,2464,3264},
    //                                                                 RGBImagedata);
    RGBImagedata = Raw2RGBEntry(Matdata,2464,3264);
    // PrintValues(RGBImagedata,0,100);
    // PrintValues(RGBImagedata,4021148,4021248);
    // PrintValues(RGBImagedata,2010524,2010624);
    /*这里必须要传递RGBImagedata这个地址的指针，如果直接传递RGBImagedata这个地址，
   那么在Raw2RGBEntry函数里面的 OutImg = convertUnsignedCharToFloat(dw_img5,out_width,out_height);
   这种做法没有改变 RGBImagedata 的值，因为在 C++ 中函数传递的是值的拷贝，OutImg 的变化不会影响到外部的 RGBImagedata。*/

    // Matdata  = Tensor2Mat(RGBImagedata,3,2464,3264);
    // cv::Mat rgbImage(2464,3264,CV_8UC3,Matdata);//rgbImage 2464行3264列 3通道无符号char的Mat类
     cv::Mat rgbImage = Tensor2Mat(RGBImagedata, 3, 2464, 3264);
    // opencvMatPrint(rgbImage);
#else
    // 2. 完成Bayer到RGB的插值
    cv::Mat rgbImage = ConvertBayer2RGB(rawImage);
#endif
#ifdef MyAWB
    uchar* Matdata = rgbImage.data;  //临时数据存储
    std::vector<float> floatRGB = Mat2Tensor(Matdata,3,row,col);


    AWBprocess(floatRGB,row,col);
    cv::Mat AWBImage = Tensor2Mat(floatRGB.data(), 3, 2464, 3264);
#else
    // 3. 对图像进行白平衡处理
    cv::Mat AWBImage;
    GrayWorldAlgorithm(rgbImage, AWBImage);
#endif
    // 4. 对图像完成颜色矫正
#ifdef MyCCM
    uchar* Matdata = AWBImage.data;  //临时数据存储
    std::vector<float> floatAWB = Mat2Tensor(Matdata,3,row,col);
    assert(!CCMprocess(floatAWB,row,col));//必须返回0才正常
    cv::Mat CCMImage = Tensor2Mat(floatAWB.data(), 3, 2464, 3264);
#else
    cv::Mat CCMImage = ColorMatrixCorrect(AWBImage);
#endif

    // 5. 对图像完成Gamma矫正
#ifdef MyGamma
    uchar* Matdata = CCMImage.data;  //临时数据存储
    std::vector<float> floatCCM = Mat2Tensor(Matdata,3,row,col);
    assert(!Gammaprocess(floatCCM,row,col));//必须返回0才正常
    cv::Mat GammaImage = Tensor2Mat(floatCCM.data(), 3, 2464, 3264);
#else
    cv::Mat GammaImage = gammaTransform(CCMImage, 0.3f); // 默认Gamma值为2.2
#endif

    // 6. 输出完成ISP调整的RGB图像，用imshow来查看效果
    cv::namedWindow("Original Raw Image", cv::WINDOW_NORMAL);
    cv::imshow("Original Raw Image", rawImage);

    cv::namedWindow("RGB Image", cv::WINDOW_NORMAL);
    cv::imshow("RGB Image", rgbImage);

    cv::namedWindow("White Balanced Image", cv::WINDOW_NORMAL);
    cv::imshow("White Balanced Image", AWBImage);

    cv::namedWindow("Color Corrected Image", cv::WINDOW_NORMAL);
    cv::imshow("Color Corrected Image", CCMImage);

    cv::namedWindow("Gamma Corrected Image", cv::WINDOW_NORMAL);
    cv::imshow("Gamma Corrected Image", GammaImage);
    
    cv::waitKey(0); // 等待按键
    return 0;
}