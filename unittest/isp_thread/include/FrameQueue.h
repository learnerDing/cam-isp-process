#ifndef FRAMEQUEUE_H
#define FRAMEQUEUE_H

#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <memory> // 包含智能指针头文件
#include <condition_variable>

extern "C" {
#include <libavutil/frame.h>
}

template <typename T>
class FrameQueue {
public:
    FrameQueue(size_t maxSize = 10); // 默认最大缓冲区数量为10
    ~FrameQueue();

    void addFrame(std::shared_ptr<T> frame);
    bool getFrame(std::shared_ptr<T>& frame);
    void stop();

private:
    // 深拷贝函数模板
    std::shared_ptr<T> deepCopy(const std::shared_ptr<T>& frame);
    std::queue<std::shared_ptr<T>> m_frameQueue; // 帧队列
    std::mutex m_mutex; // 互斥锁
    std::condition_variable m_cond; // 条件变量
    bool m_running; // 是否运行
    size_t m_maxSize; // 最大缓冲区数量
};

// 特化深拷贝函数模板为AVFrame
template <>
std::shared_ptr<AVFrame> FrameQueue<AVFrame>::deepCopy(const std::shared_ptr<AVFrame>& frame);

// 特化深拷贝函数模板为cv::Mat
template <>
std::shared_ptr<cv::Mat> FrameQueue<cv::Mat>::deepCopy(const std::shared_ptr<cv::Mat>& frame);

// 显式实例化模板类
template class FrameQueue<AVFrame>;
template class FrameQueue<cv::Mat>; // 支持OpenCV

#endif // FRAMEQUEUE_H