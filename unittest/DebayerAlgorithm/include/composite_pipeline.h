// include/composite_pipeline.h
#pragma once
#include "pipeline.h"
#include <vector>
#include <memory>
//CompositePipeline:多isp pipeline的组合和排序
class CompositePipeline : public Pipeline {
public:
    // 添加一个处理步骤
    void addStep(std::unique_ptr<Pipeline> step) {
        steps_.push_back(std::move(step));
    }

    /*
    实现基类的 process 方法
    开始运行isppipeline，输入一张cv::mat,输出为cv::mat
    */ 
    cv::Mat runisppipeline(const cv::Mat& input) override {
        cv::Mat result = input.clone(); // 深拷贝输入图像
        for (auto& step : steps_) {
            result = step->process(result); // 依次执行每个步骤
        }
        return result;
    }

private:
    std::vector<std::unique_ptr<Pipeline>> steps_; // 存储所有处理步骤
};