//This is CamThread.cpp
#include "CamThread.h"
#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
CamThread::CamThread(FrameQueue<std::shared_ptr<cv::Mat>>& frameQueue, const std::string& devicePath, int width, int height, Cam::Camtype camtype)
    : m_frameQueue(frameQueue), m_devicePath(devicePath), m_width(width), m_height(height), Camtype(Cam::Camtype::V4l2), m_camera(nullptr) {
    // 创建摄像头实例
    m_camera = Cam::createCamera(m_devicePath.c_str());
    if (!m_camera) {
        throw std::runtime_error("Failed to create camera.");
    }
}

CamThread::~CamThread() {
    if (m_camera) {
        delete m_camera;
    }
}
// 将 RAW10 格式转换为 RAW8 格式的函数
void raw10_to_raw8(const uint16_t* raw10, uint8_t* raw8, int width, int height) {
    for (int i = 0; i < width * height; i++) {
        raw8[i] = raw10[i] >> 2;  // 将 10 位数据右移 2 位，转换为 8 位
    }
}

cv::Mat createMatFromRaw8(const uint8_t* raw8, int width, int height) {
    // 创建 cv::Mat 对象，类型为 CV_8UC1（8 位无符号单通道）
    cv::Mat mat(height, width, CV_8UC1, const_cast<uint8_t*>(raw8));

    // 返回深拷贝的 Mat，避免数据被释放
    return mat.clone();
}

void CamThread::run() {
    unsigned int size = 0;
    while (true) {
        // 从摄像头获取原始数据
        void* rawData = m_camera->CameraGetRaw(m_devicePath.c_str(), m_width, m_height, Cam::V4l2, &size);
        if (!rawData) {
            std::cerr << "Failed to get raw data from camera." << std::endl;
            break;
        }

        // 判断 rawbit 参数
        int rawbit = 10;  // 假设 rawbit 为 10
        uint8_t* raw8 = new uint8_t[m_width * m_height];
        if (rawbit == 10) {
            // 将 RAW10 转换为 RAW8
            raw10_to_raw8(static_cast<uint16_t*>(rawData), raw8, m_width, m_height);
        } else if (rawbit == 8) {
            // 如果已经是 RAW8，直接复制数据
            std::memcpy(raw8, rawData, m_width * m_height * sizeof(uint8_t));
        } else {
            std::cerr << "Unsupported rawbit value: " << rawbit << std::endl;
            delete[] raw8;
            delete[] static_cast<uint8_t*>(rawData);
            break;
        }

        // 将 RAW8 数据封装到 cv::Mat 中
        cv::Mat mat = createMatFromRaw8(raw8, m_width, m_height);

        // 将 cv::Mat 包装成 std::shared_ptr<cv::Mat>
        std::shared_ptr<cv::Mat> matPtr = std::make_shared<cv::Mat>(mat);

        // 将数据添加到队列中
        m_frameQueue.addFrame(matPtr);

        // 释放原始数据
        delete[] raw8;
        delete[] static_cast<uint8_t*>(rawData);

        // 打印调试信息
        std::cout << "Captured raw data of size: " << size << std::endl;
    }

    // 停止队列
    m_frameQueue.stop();
}