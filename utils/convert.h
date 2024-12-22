#pragma once
#include <iostream>
#include <cstdlib> // for std::malloc, std::free
#include <algorithm> // for std::min, std::max>

unsigned char* convertFloatToUnsignedChar(float* floatData, int channel, int width, int height);
float* convertUnsignedCharToFloat(unsigned char* ucharData, int channel, int width, int height);
template <typename T>
void PrintValues(T* ptr, int begin, int end);