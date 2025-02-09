// this is include/debayer_pipeline.h
#pragma once
#include "pipeline.h"

// Debayer抽象基类
class DebayerPipeline : public Pipeline {
public:
    static std::unique_ptr<DebayerPipeline> create(int w, int h) ;
    static std::unique_ptr<DebayerPipeline> create(bool useCuda, Tensor&& input, int w, int h);
};
