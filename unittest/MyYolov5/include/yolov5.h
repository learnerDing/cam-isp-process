// yolov5.h
#ifndef YOLOV5_H
#define YOLOV5_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <ncnn/net.h>
#define YOLOV5_V62 1
struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
};

// 添加函数声明
ncnn::Layer* YoloV5Focus_layer_creator();
int detect_yolov5(ncnn::Net& net, const cv::Mat& bgr, std::vector<Object>& objects);
void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects);
#endif // YOLOV5_H