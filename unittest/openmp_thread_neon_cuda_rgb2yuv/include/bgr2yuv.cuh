// include/bgr2yuv.cuh

#pragma once
#include "Tensor.h"

void launch_bgr2yuv(Tensor* d_bgr, Tensor* d_yuv, int width, int height);