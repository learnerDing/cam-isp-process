//Camtest.cpp
#include "Cam.h"
#include <memory>
#include <iostream>
#include <algorithm>
#include <string>
int main() {

    const char* devicePath = "/dev/video0";
    unsigned int size = 0;
    // 通过工厂函数创建相机实例
    //std::cout<<"ok3"<<std::endl;
    Cam* camera = Cam::createCamera(devicePath);
    //std::cout<<"ok4"<<std::endl;
    if (!camera) {
        std::cerr << "Failed to create camera.\n";
        return EXIT_FAILURE;
    }
    // Call the function to get raw data
   void* rawData = camera->CameraGetRaw(devicePath, 1920, 1080,Cam::V4l2,&size);
        if (rawData != nullptr) {
            std::cout<<"success to get a raw"<<std::endl;
         return EXIT_FAILURE;
         }

    delete rawData;

    return EXIT_SUCCESS;
}