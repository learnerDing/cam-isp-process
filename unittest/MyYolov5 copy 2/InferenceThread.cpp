//This is InferenceThread.cpp
#include "include/InferenceThread.h"
#include "yolov5.h" // 包含检测函数和Object声明
#include <opencv2/opencv.hpp>

InferenceThread::InferenceThread(FrameQueue<cv::Mat>& frameQueue, 
                                FrameQueue<cv::Mat>* previewQueue = nullptr)
                                : m_frameQueue(&frameQueue),
                                m_previewQueue(previewQueue)
{
    init_model();
}

InferenceThread::~InferenceThread()
{
    yolov5_net.clear();
}

void InferenceThread::init_model()
{
    yolov5_net.opt.use_vulkan_compute = true;

#if !defined(YOLOV5_V60) && !defined(YOLOV5_V62)
    yolov5_net.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
#endif

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
        throw std::runtime_error("Model load failed");
    }

    // 预热
    cv::Mat warmup_mat(640, 640, CV_8UC3, cv::Scalar(0));
    std::vector<Object> warmup_objects;
    detect_yolov5(yolov5_net, warmup_mat, warmup_objects);
}

void InferenceThread::run() 
{
    while (true) 
    {
        std::shared_ptr<cv::Mat> frame;
        if (!m_frameQueue->getFrame(frame,200)) 
        {
            break;
        }
        processFrame(*frame);
    }
}

void InferenceThread::processFrame(const cv::Mat& frame) 
{
    std::vector<Object> objects;
    if (0 == detect_yolov5(yolov5_net, frame, objects))
    {
        if (m_previewQueue) {//如果开启了预览，则向预览缓冲区队列加入缩小后的frame
            cv::Mat resultFrame = frame.clone();
            // 绘制检测框到resultFrame...
            cv::resize(resultFrame, resultFrame, cv::Size(320, 320)); // 缩小分辨率
            auto previewFrame = std::make_shared<cv::Mat>(resultFrame);
            m_previewQueue->addFrame(previewFrame);
        }
        // 打印检测结果
        std::cout << "\nDetected " << objects.size() << " objects:" << std::endl;
        for (const auto& obj : objects)
        {
            std::cout << "Label: " << obj.label 
                      << " | Prob: " << obj.prob 
                      << " | Rect: [" << obj.rect.x << ", " << obj.rect.y
                      << ", " << obj.rect.width << ", " << obj.rect.height << "]\n";
        }
    }
}
