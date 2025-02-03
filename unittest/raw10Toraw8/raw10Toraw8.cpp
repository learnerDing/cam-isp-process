#include <iostream>
#include <cstdint>
#include <chrono>
#include <cstdlib>
#include <omp.h>
#include <arm_neon.h>

using namespace std;
/*
编译方式：g++ -O3 -fopenmp -march=native -o raw10Toraw8 raw10Toraw8.cpp
*/

// 使用NEON+OpenMP优化的转换函数
void raw10_to_raw8(uint16_t* __restrict raw10, uint8_t* __restrict raw8, int width, int height) {
    const int total = width * height;
    const int chunk_size = 64; // 每个NEON块处理64个元素（8x8）

    #pragma omp parallel for
    for (int i = 0; i < total; i += chunk_size) {
        const int end_idx = min(i + chunk_size, total);
        int j = i;
        
        // NEON处理（每次处理8个元素）
        for (; j <= end_idx - 8; j += 8) {
            uint16x8_t in = vld1q_u16(raw10 + j);
            uint16x8_t shifted = vshrq_n_u16(in, 2);
            uint8x8_t out = vmovn_u16(shifted);
            vst1_u8(raw8 + j, out);
        }
        
        // 处理剩余元素
        for (; j < end_idx; ++j) {
            raw8[j] = raw10[j] >> 2;
        }
    }
}

int main() {
    const int width = 1920;
    const int height = 1080;
    const int total_pixels = width * height;
    const int iterations = 100;

    // 使用对齐内存分配（16字节对齐）
    uint16_t* raw10 = static_cast<uint16_t*>(aligned_alloc(16, total_pixels * sizeof(uint16_t)));
    uint8_t* raw8 = static_cast<uint8_t*>(aligned_alloc(16, total_pixels * sizeof(uint8_t)));

    // 生成模拟RAW10数据（10位有效数据）
    #pragma omp parallel for
    for (int i = 0; i < total_pixels; ++i) {
        raw10[i] = static_cast<uint16_t>(i & 0x3FF); // 取低10位
    }

    // 预热缓存（避免冷启动误差）
    raw10_to_raw8(raw10, raw8, width, height);

    // 正式性能测试
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        raw10_to_raw8(raw10, raw8, width, height);
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = end - start;

    cout << "转换次数: " << iterations << endl;
    cout << "总耗时: " << elapsed.count() << " 秒" << endl;
    cout << "单次平均耗时: " << elapsed.count()/iterations << " 秒" << endl;

    // 释放内存
    free(raw10);
    free(raw8);

    return 0;
}