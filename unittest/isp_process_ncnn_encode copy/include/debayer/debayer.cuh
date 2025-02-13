// include/debayer.cuh

#pragma once
#include "Tensor.h"

void launchDebayer(const Tensor& input, Tensor& output);