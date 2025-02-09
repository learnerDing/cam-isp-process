//This is debayer_ocv_pipeline.h
#pragma once
#include "debayer_pipeline.h"
// OpenCV实现
class DebayerOCVPipeline : public DebayerPipeline {
public:
    cv::Mat process(const cv::Mat& input) override;
};