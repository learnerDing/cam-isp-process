// This is src/debayer_ocv.cpp
#include "../include/debayer_ocv_pipeline.h"

cv::Mat DebayerOCVPipeline::process(const cv::Mat& input) {
    cv::Mat output;
    cv::cvtColor(input, output, cv::COLOR_BayerRG2BGR);
    return output;
}
