//This is CamThread.h
#ifndef CAMTHREAD_H
#define CAMTHREAD_H

#include <string>
#include <memory>
#include <opencv2/core.hpp>
#include "FrameQueue.h"
#include "Cam.h"
#include "Thread.h"  // 包含自定义的 Thread 头文件

class CamThread : public Thread
{
public:
    CamThread(
        FrameQueue<std::shared_ptr<cv::Mat>>& frameQueue,
        const std::string& devicePath,
        int width,
        int height,
        Cam::Camtype camtype
    );
    ~CamThread();

protected:
    void run() override;  // 覆盖 Thread 的纯虚函数

private:
    FrameQueue<std::shared_ptr<cv::Mat>>& m_frameQueue;
    std::string m_devicePath;
    int m_width;
    int m_height;
    Cam::Camtype m_camType;  // 修正成员变量名以匹配初始化列表
    Cam* m_camera;

    // RAW 格式转换工具函数
    void raw10_to_raw8(const uint16_t* raw10, uint8_t* raw8, int width, int height);
    cv::Mat createMatFromRaw8(const uint8_t* raw8, int width, int height);
};

#endif // CAMTHREAD_H