//This is Raw8Provider.cpp
/*raw8图像流提供线程：
使用5张图像轮流定时加载送入FrameQueue模拟1920x1080x30fps的视频流
*/
#include <fstream>
#include <iostream>
#include "Raw8Provider.h"

Raw8Provider::Raw8Provider(FrameQueue<cv::Mat>& frameQueue)
    : m_frameQueue(frameQueue), m_stop(false) {
    loadRawImages();
}

Raw8Provider::~Raw8Provider() {
    stop();
}

void Raw8Provider::stop() {
    m_stop = true;
}

void Raw8Provider::loadRawImages() {
    const int width = 1920;
    const int height = 1080;
    const int expectedSize = width * height;

    for (int i = 1; i <= 5; ++i) {
        std::string filename = "../rawprovide/1920_1080_8_" + std::to_string(i) + ".raw";
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            continue;
        }

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        if (size != expectedSize) {
            std::cerr << "Invalid file size for " << filename 
                      << " (expected " << expectedSize 
                      << ", got " << size << ")" << std::endl;
            file.close();
            continue;
        }

        std::vector<uchar> buffer(size);
        if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
            std::cerr << "Error reading file: " << filename << std::endl;
            file.close();
            continue;
        }
        file.close();

        auto img = std::make_shared<cv::Mat>(height, width, CV_8UC1);
        memcpy(img->data, buffer.data(), size);
        m_images.push_back(img);
    }

    if (m_images.empty()) {
        std::cerr << "No valid images loaded!" << std::endl;
    }
}

void Raw8Provider::run() {
    using namespace std::chrono;
    size_t index = 0;
    
    while (!m_stop) {
        auto start_time = steady_clock::now();

        if (!m_images.empty()) {
            m_frameQueue.addFrame(m_images[index]);
            index = (index + 1) % m_images.size();
        }

        auto frame_duration = milliseconds(35);
        auto elapsed_time = steady_clock::now() - start_time;
        auto sleep_time = frame_duration - duration_cast<milliseconds>(elapsed_time);

        if (sleep_time.count() > 0) {
            std::this_thread::sleep_for(sleep_time);
        }
    }
}