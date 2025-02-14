//This is main.cpp
#include "Raw8Provider.h"
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <csignal>

// 全局标志位用于安全退出
std::atomic<bool> g_running(true);

// 信号处理函数
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    g_running = false;
}

int main() {
    // 注册信号处理
    std::signal(SIGINT, signalHandler);

    const size_t queue_size = 10;
    FrameQueue<cv::Mat> frame_queue(queue_size);

    try {
        // 创建生产者
        Raw8Provider provider(frame_queue);
        
        //
 启动生产者线程
        provider.start();
        std::cout << "Start streaming... Press Ctrl+C to exit\n";

        std::shared_ptr<cv::Mat> frame_ptr;
        const std::string window_name = "Raw8 Viewer";

        while (g_running) {
            
            if (frame_queue.getFrame(frame_ptr,200)) { 
                // 显示图像
                cv::imshow(window_name, *frame_ptr);
                
                // 控制显示速率（约30fps）
                int key = cv::waitKey(33);
                if (key == 27) { // ESC键退出
                    g_running = false;
                }
            } else {
                std::cout << "Frame queue timeout, retrying...\n";
            }
        }

        // 停止生产者
        provider.stop();
        provider.join();

    } catch (const std::exception& e) {
        std::cerr << "Error occurred: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "Application exited safely.\n";
    return EXIT_SUCCESS;
}
