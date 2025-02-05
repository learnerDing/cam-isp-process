//This is Cam.h
#ifndef CAM_H
#define CAM_H

#include <iostream>
#include <fcntl.h>
#include <cstring>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <vector>
#include <string>
#include <memory>
// #define REQBUFS_COUNT 4    // Number of buffers

class Cam {
public:
    enum Camtype
    {
        V4l2 = 0
    };
    //virtual Cam() = default;
    virtual ~Cam() = default;
    static Cam* createCamera(const std::string& devicePath,Camtype type=V4l2);//创建具体的摄像头对象
    // virtual int initialize(int width, int height) = 0;
    // virtual void* captureRawData(unsigned int* size) = 0;
    // virtual int stopCamera() = 0;
    // virtual int exitCamera() = 0; 
    // 工厂函数声明，在Cam.cpp里面分发为具体的摄像头控制函数
    void* CameraGetRaw(const std::string& devicePath, int width, int height,Camtype type,unsigned int* sizeptr);
     //具体的拍摄方法，在v4l2或者其他类中重写
    virtual void* GetRaw(const std::string& devicePath, int width, int height,unsigned int* sizeptr)=0;
   
protected:
    int fd;
    // cam_buf bufs[REQBUFS_COUNT];
    //struct v4l2_requestbuffers reqbufs;

    // virtual int getCapability() = 0;
    // virtual int getSupportedVideoFormats() = 0;
    // virtual int setVideoFormat(int width, int height) = 0;
    // virtual int requestBuffers() = 0;
    // virtual int startCamera() = 0;
    // virtual int dqBuffer(void** buf, unsigned int* size, unsigned int* index) = 0;
    // virtual int eqBuffer(unsigned int index) = 0;
};

//extern "C" void* CameraGetRaw(const char* devicePath, int width, int height);
#endif // CAM_H