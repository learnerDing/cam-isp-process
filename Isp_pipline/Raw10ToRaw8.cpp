#include "Raw10ToRaw8.h"
#include <iostream> // 用于日志或输出

void Raw10ToRaw8::process(const Image& input, Image& output) {
    std::cout << "Processing Raw10 to Raw8..." << std::endl;
    
#ifdef USE_CUDA
    // 调用CUDA版本处理
    // Raw10ToRaw8_CUDA(input, output);
#else
    // ARM纯软件实现
    // Raw10ToRaw8_arm(input, output);
#endif

#ifdef USE_NEON
    // 若使用NEON加速的逻辑
#else
    // 常规处理
#endif

#ifdef USE_OPENMP
    #pragma omp parallel for
    for (int i = 0; i < input.height(); i++) {
        // 行处理逻辑
    }
#else
    // 单线程处理
#endif
}