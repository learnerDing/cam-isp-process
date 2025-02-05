#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "Tensor.h"
// #include "src/utils/vectorize_utils.h"

template<typename T>
void launch_raw10_to_raw8_cuda(TensorWrapper<T>* d_raw10, int rows, int cols);