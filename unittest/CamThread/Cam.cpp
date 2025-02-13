//This is Cam.cpp
#include "Cam.h"
#include "v4l2cam.h"  // 只包含具体实现的头文件

// 工厂模式分发器
std::unique_ptr<Cam> Cam::Create(Type type, const std::string& device_path, FrameQueue<cv::Mat>* frameQueue) {
    switch(type) {
        case Type::V4L2:
            return std::unique_ptr<Cam>(new V4L2Camera(device_path, frameQueue));
        default:
            throw std::invalid_argument("Unsupported camera type");
    }
}
// 使用NEON+OpenMP优化的高性能转换函数
void raw10_to_raw8(uint16_t* __restrict raw10, uint8_t* __restrict raw8, int width, int height) {
    const int total = width * height;
    const int chunk_size = 64; // 每个NEON块处理64个元素（8x8）

    #pragma omp parallel for
    for (int i = 0; i < total; i += chunk_size) {
        const int end_idx = std::min(i + chunk_size, total);
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