#ifndef STREAM_H
#define STREAM_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <opencv2/opencv.hpp> // 确保已链接 OpenCV 库

class Stream {
public:
    Stream();
    ~Stream();

    void addFrame(const cv::Mat& frame);
    bool getFrame(cv::Mat& frame);
    void stop();

private:
    std::queue<cv::Mat> m_frameQueue; // 帧队列
    std::mutex m_mutex; // 互斥锁
    std::condition_variable m_cond; // 条件变量
    bool m_running; // 是否运行
};

#endif // STREAM_H