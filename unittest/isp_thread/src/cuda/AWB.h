#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "Tensor.h"


template <typename T>
void launchAWB(TensorWrapper<T>* d_img_tensor, int rows, int cols);