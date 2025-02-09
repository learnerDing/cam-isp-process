// 基类设计 (核心架构)
// This is include/pipeline.h
#pragma once
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>
#include "composite_pipeline.h"
#include "Tensor.h"
class Pipeline {
public:
    virtual ~Pipeline() = default;
    
    // 统一处理接口
    virtual cv::Mat runisppipeline(const cv::Mat& input) = 0;
    
    // 工厂方法
    static std::unique_ptr<Pipeline> create(bool useCuda,const int width = 1920,const int height = 1080);
};


class PipelineStep : public Pipeline {
protected:
    std::unique_ptr<Pipeline> nextStep;

public:
    void setNextStep(std::unique_ptr<Pipeline> step) {
        nextStep = std::move(step);
    }
};
