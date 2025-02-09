// 工厂实现
// This is src/pipeline.cpp
#include "pipeline.h"
#include "debayer_pipeline.h"

std::unique_ptr<Pipeline> Pipeline::create(bool useCuda,const int width,const int height) {
    // 创建完整处理链
    auto compose_pipeline = std::make_unique<CompositePipeline>();
    // 添加处理步骤
    if(useCuda)
    {
    Tensor dummy_input(...);
    compose_pipeline->addStep(DebayerPipeline::create(useCuda, std::move(dummy_input), 1920, 1080));
    // compose_pipeline->addStep(AWBPipeline::create(useCuda));
    // compose_pipeline->addStep(CCMPipeline::create(useCuda));
    // compose_pipeline->addStep(GammaPipeline::create(useCuda));
    }
    else
    {
        compose_pipeline->addStep(DebayerPipeline::create(1920,1080));
    }


    
    return compose_pipeline;
}