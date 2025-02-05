// This is RawScalar.cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifndef OpenCV
//#include "BasicBitmap.h"
#include "bitmap_image.hpp"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define openmp
#ifdef openmp
#include <omp.h>
#endif
#endif // !OpenCV
#include "rawscaler.h"
#include "RawScaler.h"
#include <math.h>
#include <time.h>  
using namespace RawScaler;  
#define ratio 1  //图像缩放倍数
// BayerType mBayerType = BGGR; //GRBG,BGGR,RGGB,GBRG
BayerType mBayerType = RGGB;
#define BOUND(a,min_val,max_val)           ( (a < min_val) ? min_val : (a >= max_val) ? (max_val) : a )

void Crop_Area_Image(unsigned char *i_data, unsigned char  *o_data, int width, int st_x, int st_y, int w, int h)
{
	int w_line_org = width * 3;
	int w_line = w * 3;
	int i_pos, o_pos;

	o_pos = 0;
	i_pos = st_y*w_line_org + st_x * 3;
	for (int y = 0; y<h; y++)
	{
		memcpy(o_data + o_pos, i_data + i_pos, sizeof(unsigned char)*w_line);
		i_pos += w_line_org;
		o_pos += w_line;
	}

}
/*添加边界：图像处理中的一种常见技巧，
通过增加边缘像素的上下文信息来提高处理效果，使得缩放后的结果更加精细和自然。
最后在获得最终图像后，再裁剪掉多余的边界，从而得到所需的图像尺寸和内容。*/
unsigned char* MakeRawBorder(unsigned char* img1, int pad_width, int width, int height)
{
	int i, j;
	unsigned char* img2 = new unsigned char[(width + pad_width * 2) * (height + pad_width * 2)];
	//#pragma omp parallel for
	for (i = 0; i < height; i++) {
		const unsigned char* ptr1 = &img1[i*width];
		unsigned char* ptr2 = &img2[(i + pad_width)*(width + pad_width * 2)];
		int img_index1 = 0;
		int img_index2 = pad_width;
		for (j = 0; j < width; j++) {

			if (j == 0) {
				for (int k = 0; k < pad_width + 1; k++) {
					if (k % 2 == 0) {
						ptr2[img_index2 - k] = ptr1[img_index1];
					}
					else {
						ptr2[img_index2 - k] = ptr1[img_index1 + 1];
					}
				}
			}
			else if (j == width - 1) {
				for (int k = 0; k < pad_width + 1; k++) {

					if (k % 2 == 0) {
						ptr2[img_index2 + k] = ptr1[img_index1];
					}
					else {
						ptr2[img_index2 + k] = ptr1[img_index1 - 1];
					}
				}
			}
			else {
				ptr2[img_index2] = ptr1[img_index1];
			}
			img_index1++;
			img_index2++;
		}
	}
	for (i = 0; i < pad_width; i++) {
		const unsigned char* ptr1;
		if (i % 2 == 0) {
			ptr1 = &img2[(pad_width + 1)*(width + pad_width * 2)];
		}
		else {
			ptr1 = &img2[(pad_width)*(width + pad_width * 2)];
		}

		unsigned char* ptr2 = &img2[(pad_width - 1 - i)*(width + pad_width * 2)];
		int img_index1 = 0;
		int img_index2 = 0;
		for (j = 0; j < width + pad_width * 2; j++) {
			ptr2[img_index2] = ptr1[img_index1];
			img_index1++;
			img_index2++;
		}
	}
	for (i = 0; i < pad_width; i++) {
		const unsigned char* ptr1;
		if (i % 2 == 0) {
			ptr1 = &img2[(height + pad_width - 2)*(width + pad_width * 2)];
		}
		else {
			ptr1 = &img2[(height + pad_width - 1)*(width + pad_width * 2)];
		}
		unsigned char* ptr2 = &img2[(pad_width + height + i)*(width + pad_width * 2)];
		int img_index1 = 0;
		int img_index2 = 0;
		for (j = 0; j < width + pad_width * 2; j++) {
			ptr2[img_index2] = ptr1[img_index1];
			img_index1++;
			img_index2++;
		}
	}
	//eqm /= height * width;
	printf("Endof MakeRawborder!");
	return img2;
}
unsigned char* RGB2RAW(bitmap_image src, BayerType type = BGGR)
{
	int w = src.get_width();
	int h = src.get_height();
	//#pragma omp parallel for
	unsigned char* int_img = new unsigned char[w*h];
	for (int i = 0; i < h; i++) {
		int img_index1 = 0;
		int img_index2 = i*w;
		int offset = i % 2;
		for (int j = 0; j < w; j++)
		{
			int ch = (j) % 2;
			//IUINT32 clr = src->GetPixel(j, i);
			rgb_t t = src.get_pixel(j, i);
			//int red = clr & 0xff;
			//int green = (clr >> 8) & 0xff;
			//int blue = (clr >> 16) & 0xff;
			if (!offset) {
				switch (type)
				{
				case RGGB:
					if (!ch) {
						int_img[img_index2] = t.red;
					}
					else {
						int_img[img_index2] = t.green;
					}
					break;
				case GRBG:
					if (!ch) {
						int_img[img_index2] = t.green;
					}
					else {
						int_img[img_index2] = t.red;
					}
					break;
				case BGGR:
					if (!ch) {
						int_img[img_index2] = t.blue;
					}
					else {
						int_img[img_index2] = t.green;
					}
					break;
				case GBRG:
					if (!ch) {
						int_img[img_index2] = t.green;
					}
					else {
						int_img[img_index2] = t.blue;
					}
					break;
				default:
					printf("error bayer type\n");
					//return raw_img;
				}

			}
			else {
				switch (mBayerType)
				{
				case RGGB:
					if (!ch) {
						int_img[img_index2] = t.green;
					}
					else {
						int_img[img_index2] = t.blue;
					}
					break;
				case GRBG:
					if (!ch) {
						int_img[img_index2] = t.blue;
					}
					else {
						int_img[img_index2] = t.green;
					}
					break;
				case BGGR:
					if (!ch) {
						int_img[img_index2] = t.green;
					}
					else {
						int_img[img_index2] = t.red;
					}
					break;
				case GBRG:
					if (!ch) {
						int_img[img_index2] = t.red;
					}
					else {
						int_img[img_index2] = t.green;
					}
					break;
				default:
					printf("error bayer type\n");
					//return raw_img;
				}

			}
			//float l = pow(ptr2[img_index2]/255.0,  2.2f);
			//ptr2[img_index2] = l * 255;
			img_index2++;
			img_index1 += 3;
		}
	}
	return int_img;
}
unsigned char* Raw2RGBEntry(unsigned char* rawdata,int rows ,int cols)
{	
	int In_width = cols;
	int In_height = rows;
	printf("The image width:%d,The height:%d\n",In_width,In_height);//kodim19.bmp: PC bitmap, Windows 3.x format, 512 x 768 x 24
	printf("size of raw data:%d\n",In_width * In_height);//size of raw data:1179648
	int pad = 6;
	int pad_w = pad * 2;//两边一共padding12
	unsigned char* pad_img = MakeRawBorder(rawdata, pad,In_width,In_height);
	//以下四步在scaling
	int out_height = In_height / ratio;
	int out_width = In_width / ratio;
	out_width = out_width + (out_width % 2);
	out_height = out_height + (out_height % 2);
	out_width += pad_w;
	out_height += pad_w;
	RawProcessor scaler;
	unsigned char* dw_img4 = new unsigned char[out_width*out_height*3];//接收Bayer转BGR的三通道图像
	unsigned char* dw_img5 = new unsigned char[(out_width-pad_w)*(out_height - pad_w) * 3];//然后去掉边界padding的三通道图像
	//void BayerToRGB(unsigned char* in, unsigned char* out, int iwidth, int iheight, int owidth, int oheight, BayerType type);
	scaler.BayerToRGB(pad_img, dw_img4, In_width+ pad_w, In_height+ pad_w, out_width, out_height, mBayerType);
	//注意：这里输出的dw_img5图像其实是HWC排列，每个像素颜色顺序是BGR
	Crop_Area_Image(dw_img4, dw_img5, out_width, pad, pad, out_width - pad_w, out_height - pad_w);//裁剪掉之前加上的边框存在dw_img5

	out_width -= pad_w;
	out_height -= pad_w;
	return dw_img5;
	delete dw_img4;
	delete pad_img;
}