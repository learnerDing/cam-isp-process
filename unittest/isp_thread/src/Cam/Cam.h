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
    enum class Type { V4L2 };
    virtual ~Cam() = default;
    
    virtual void* CaptureFrame(int width, int height, unsigned int* out_size) = 0;
    virtual void ReleaseFrame(void* frame) = 0;

    static std::unique_ptr<Cam> Create(Type type, const std::string& device_path);
protected:
    Cam() = default;
};
#endif // CAM_H