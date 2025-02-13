// This is CamTest.cpp
#include "Cam.h"
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
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
            if(raw_format == "raw10") {   
                uint16_t* raw10frame = static_cast<uint16_t*>(frame_data);
                uint8_t* raw8frame = static_cast<uint8_t*>(aligned_alloc(16, width * height * sizeof(uint8_t)));
                
                // 转换RAW10到RAW8
                raw10_to_raw8(raw10frame, raw8frame, width, height);
                
                // 保存RAW8文件
                char filename[256];
                snprintf(filename, sizeof(filename), "%d_%d_raw8.raw", width, height);
                FILE* file = fopen(filename, "wb");
                if(file) {
                    fwrite(raw8frame, 1, width * height, file);
                    fclose(file);
                    std::cout << "Saved RAW8 image to: " << filename << std::endl;
                } else {
                    std::cerr << "Failed to save file: " << filename << std::endl;
                }
                
                free(raw8frame);  // 释放RAW8缓冲区
            }
        }
        
        camera->ReleaseFrame(frame_data);
        return EXIT_SUCCESS;
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}