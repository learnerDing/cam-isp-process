// include/rgb2yuv.cuh

#pragma once
#include "Tensor.h"

void launch_rgb2yuv(Tensor* d_rgb, Tensor* d_yuv, int width, int height);