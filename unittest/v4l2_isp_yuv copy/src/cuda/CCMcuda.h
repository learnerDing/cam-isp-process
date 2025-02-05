#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "Tensor.h"


template <typename T>
void launchCCMGemm(TensorWrapper<T>* d_img_tensor, TensorWrapper<T>* CCM_mat_tensor ,int width, int height);