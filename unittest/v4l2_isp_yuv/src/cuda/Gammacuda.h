#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "Tensor.h"


template <typename T>
void launchGammaCorrection(TensorWrapper<T>* d_img_tensor, TensorWrapper<T>* d_lut_tensor, int rows, int cols);