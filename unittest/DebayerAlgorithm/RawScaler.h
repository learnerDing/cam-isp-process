#ifndef RAW_SCALER_H
#define RAW_SCALER_H

#include <stdio.h>

#include "utils/convert.h"
#include "bitmap_image.hpp"
#include "rawscaler.h"

#ifdef openmp
#include <omp.h>
#endif

using namespace RawScaler;

#define RATIO 1  // 图像缩放倍数
#define BOUND(a, min_val, max_val) ((a < min_val) ? min_val : (a >= max_val) ? (max_val) : a)


void Crop_Area_Image(unsigned char *i_data, unsigned char *o_data, int width, int st_x, int st_y, int w, int h);
unsigned char* MakeRawBorder(unsigned char* img1, int pad_width, int width, int height);
unsigned char* RGB2RAW(bitmap_image src, BayerType type);
unsigned char* Raw2RGBEntry(unsigned char* rawdata,int rows ,int cols);

#endif // RAW_SCALER_H