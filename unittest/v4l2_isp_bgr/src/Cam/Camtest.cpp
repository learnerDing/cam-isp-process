// CamTest.cpp
#include "Cam.h"
#include <iostream>
#include <omp.h>
#include <arm_neon.h>
#include <cstdlib>
#include <cstdint>
#include <opencv2/opencv.hpp>

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
int main(int argc,char* argv[]) {
    try {
        int width = 1920;
        int height = 1080;
        std::string raw_format = "raw10";
        if(argc >= 4) {
            width = std::stoi(argv[1]);
            height = std::stoi(argv[2]);
            raw_format = argv[3];
        }
        auto camera = Cam::Create(Cam::Type::V4L2, "/dev/video0");
        
        unsigned int frame_size = 0;
        void* frame_data = camera->CaptureFrame(width, height, &frame_size);

        if(frame_data && frame_size > 0) {
            std::cout << "Captured frame: " << frame_size << " bytes\n";
            if(raw_format=="raw10")
        {   uint16_t* raw10frame = static_cast<uint16_t*>(frame_data);//raw10一个像素点占两个字节
            uint8_t* raw8frame = static_cast<uint8_t*>(aligned_alloc(16, width*height * sizeof(uint8_t)));//16字节对其
            raw10_to_raw8(raw10frame,raw8frame,width,height);
            }
        }
        
        camera->ReleaseFrame(frame_data);
        return EXIT_SUCCESS;
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}