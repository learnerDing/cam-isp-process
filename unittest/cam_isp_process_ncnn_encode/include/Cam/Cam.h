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
#include "../util/FrameQueue.h"
// #define REQBUFS_COUNT 4    // Number of buffers

void raw10_to_raw8(uint16_t* __restrict raw10, uint8_t* __restrict raw8, int width, int height);
class Cam {
public:
    enum class Type { V4L2 };
    virtual ~Cam() = default;
    
    virtual void Start(int width, int height, const std::string& raw_format) = 0;
    virtual void Stop() = 0;

    static std::unique_ptr<Cam> Create(Type type,
                                        const std::string& device_path, 
                                        FrameQueue<cv::Mat>* frameQueue);

protected:
    explicit Cam(FrameQueue<cv::Mat>* frameQueue) : m_frameQueue(frameQueue) {}
    FrameQueue<cv::Mat>* m_frameQueue;
};
#endif // CAM_H

//工程模式管理相机对象
// 当需要新增相机类型时：
// 1. 在Cam::Type枚举中添加新类型
// 2. 在Cam::Create工厂方法中添加对应case
// 3. 实现新的派生类（如UvcCamera），继承Cam并实现接口
// 4. 客户端代码无需修改即可通过工厂创建新类型