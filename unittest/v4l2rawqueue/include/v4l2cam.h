// v4l2cam.h
#ifndef V4L2CAM_H
#define V4L2CAM_H

#include "Cam.h"
#include <linux/videodev2.h>
#include <cstring>
#define REQBUFS_COUNT 4    // Number of buffers

struct cam_buf {
    void *start;
    size_t length;
};

class V4L2Camera : public Cam {
public:
    V4L2Camera(const std::string& devicePath);
    ~V4L2Camera() override;

    int initialize(int fd,int width, int height);
    void* captureRawData(int fd,unsigned int* size) ;
    int stopCamera(int fd);
    int exitCamera(int fd) ;
    // int width(){return cam_width_;}
    // int height(){return cam_height_;}
    void setwidth(int width){cam_width_=width;}
    void setheight(int height){cam_height_=height;}
    void setpathdevice(const std::string& devicepath ="/dev/video0"){devicepath_,devicepath;}
    void* GetRaw(const std::string &devpath,int width,int height,unsigned int* sizeptr)override;
    //void* CameraGetRaw(const char* devicePath,int width,int height) override;
    cam_buf bufs[REQBUFS_COUNT];
protected:
    //cam_buf bufs[REQBUFS_COUNT];
    int cam_width_ = 0;
    int cam_height_ = 0;
    char* devicepath_ = "/dev/video0";
    unsigned int * imgsize_ = 0; 
    struct v4l2_requestbuffers reqbufs;
    int getCapability(int fd) ;
    int getSupportedVideoFormats(int fd) ;
    int setVideoFormat(int fd,int width, int height) ;
    int requestBuffers(int fd) ;
    int startCamera(int fd) ;
    int dqBuffer(int fd,void** buf, unsigned int* size, unsigned int* index) ;
    int eqBuffer(int fd,unsigned int index) ;
};
//extern "C" void* CameraGetRaw(const char* devicePath,int width,int height);
#endif