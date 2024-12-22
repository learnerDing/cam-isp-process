#ifndef ISPIPELINE_H
#define ISPIPELINE_H

#include "Image.h" // 假设你有一个 Image 类来表示图像
//IspPipeline :isp图像处理基类
class ISPPipeline {
public:
    virtual void process(const Image& input, Image& output) = 0;
    virtual ~ISPPipeline() {}
};

#endif // ISPIPELINE_H