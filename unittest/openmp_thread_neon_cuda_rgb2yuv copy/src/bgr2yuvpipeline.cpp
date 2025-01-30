// this is src/bgr2yuvpipeline.cpp
#include "bgr2yuvpipeline.h"
#include "bgr2yuv.cuh"
#define DBG_BGR2YUV
BGR2YUVPipeline::BGR2YUVPipeline(Tensor&& InTensor_cpu, 
                                int width, int height) 
    : width_(width), height_(height) {
    // 输入检查
    assert(InTensor_cpu.device() == DeviceType::CPU);
     // 新增尺寸校验
    if (width%2 !=0 || height%2 !=0) {
        throw std::invalid_argument("Width/height must be even for YUV420P");
    }
    #ifdef DBG_BGR2YUV
        printf("Pipeline CTOR: input tensor ptr=%p\n", InTensor_cpu.data());
    #endif
    // 创建GPU Tensor
    bgr_tensor_gpu_ = InTensor_cpu.cputogpu();
    //yuv420p分配空间需要额外处理
    // 修改YUV420P的Tensor形状
    const int y_size = width * height;
    const int uv_size = (width/2) * (height/2);
 // yuv420p平铺存储 格式为[1,1,1.5*width*height]
    std::vector<int> yuv_shape = {1, 1, y_size + 2*uv_size};
    yuv_tensor_gpu_ = Tensor(yuv_shape, DataType::FLOAT32, DeviceType::GPU);

}

BGR2YUVPipeline::~BGR2YUVPipeline() {
    #ifdef DBG_BGR2YUV
        printf("~BGR2YUVPipeline()\n");
    #endif
}

Tensor BGR2YUVPipeline::process() {
    // 执行CUDA核函数
     bgr_tensor_gpu_.print(0,0,4,"bgr_tensor_gpu_ before lauch cuda print");
    launch_bgr2yuv(
        &bgr_tensor_gpu_,&yuv_tensor_gpu_,
        width_, height_
    );

    yuv_tensor_gpu_.print(0,0,24,"yuv_tensor_gpu_ after lauch cuda print");
    //printf("width_=%d,height_=%d\n",width_,height_);
    // 返回CPU Tensor，右值引用不拷贝
    return yuv_tensor_gpu_.gputocpu();//这里创建的shape也是[1,1,1.5*width*height]的yuv420p的shape
}