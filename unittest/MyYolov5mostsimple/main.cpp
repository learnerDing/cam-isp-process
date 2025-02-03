// main.cpp
#include "yolov5.h"
#include <chrono>
#include <iostream>
#include <vector>
#define YOLOV5_V62 1
int main(int argc, char** argv)
{
    // 检查命令行参数
    if (argc != 2)
    {
        std::cerr << "Usage: " << argv[0] << " <imagepath>\n";
        return -1;
    }

    // 读取图像
    const char* imagepath = argv[1];
    cv::Mat image = cv::imread(imagepath);
    if (image.empty())
    {
        std::cerr << "Failed to read image: " << imagepath << "\n";
        return -1;
    }

    // 初始化模型（只需一次）
    ncnn::Net yolov5_net;
    yolov5_net.opt.use_vulkan_compute = true;

// 根据模型版本注册自定义层
#if !defined(YOLOV5_V60) && !defined(YOLOV5_V62)
    yolov5_net.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
#endif

// 加载模型参数和权重
#if YOLOV5_V62
    if (yolov5_net.load_param("yolov5s_6.2.param") ||
        yolov5_net.load_model("yolov5s_6.2.bin"))
#elif YOLOV5_V60
    if (yolov5_net.load_param("yolov5s_6.0.param") ||
        yolov5_net.load_model("yolov5s_6.0.bin"))
#else
    if (yolov5_net.load_param("yolov5s.param") ||
        yolov5_net.load_model("yolov5s.bin"))
#endif
    {
        std::cerr << "Failed to load YOLOv5 model\n";
        return -1;
    }

    // 预热（可选）
    std::vector<Object> warmup_objects;
    detect_yolov5(yolov5_net, image, warmup_objects);

    // 正式计时测试
    const int num_detections = 100;
    std::vector<Object> objects;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_detections; ++i)
    {
        objects.clear();
        detect_yolov5(yolov5_net, image, objects);
        
        // 打印每次检测结果（可选）
        std::cout << "Detection " << i+1 << " found " << objects.size() << " objects\n";
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // 输出计时结果
    std::cout << "\n=== Timing Results ===\n";
    std::cout << "Total time for " << num_detections << " detections: "
              << duration.count() << " ms\n";
    std::cout << "Average time per detection: " 
              << duration.count() / static_cast<double>(num_detections) << " ms\n";

    return 0;
}