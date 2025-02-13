// This is CamTest.cpp
#include "Cam.h" 
#include <iostream>
#include <cstdlib>
#include <cstdint>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
    try {
        int width = 1920;
        int height = 1080;
        std::string raw_format = "raw10";
        if(argc >= 4) {
            width = std::stoi(argv[1]);
            height = std::stoi(argv[2]);
            raw_format = argv[3];
        }
        FrameQueue<cv::Mat> frameQueue(5, 1); // 队列容量30，满时丢弃2旧帧
        auto camera = Cam::Create(Cam::Type::V4L2, "/dev/video0" ,&frameQueue);
        
 // 启动采集线程
        camera->Start(width, height, raw_format);
        
        // 消费者显示循环
        cv::namedWindow("Preview", cv::WINDOW_AUTOSIZE);
        while (true) {
            std::shared_ptr<cv::Mat> frame;
            if (frameQueue.getFrame(frame)) {
                cv::imshow("Preview", *frame);
                if (cv::waitKey(1) == 27) break; // ESC退出
            }
        }
        
        camera->Stop();
        return EXIT_SUCCESS;
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}