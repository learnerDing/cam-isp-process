//This is Tensor.h
#ifndef _TENSOR_H
#define _TENSOR_H
#include <vector>
#include <cassert>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <cuda_runtime.h>
#define DBG_Tensor
enum class DeviceType { CPU, GPU };
enum class DataType { UINT8, FLOAT32 };
// #define USE_CUDA
class Tensor {
private:
    void* data_ptr_ = nullptr;
    DeviceType device_ = DeviceType::CPU;
    DataType dtype_ = DataType::FLOAT32;
    std::vector<int> shape_;

public:
    int width_ = 1920;//图像宽度，和数据形状无关
    int height_ = 1080;//图像高度，和数据形状无关

    // 删除拷贝构造和拷贝赋值
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // 移动构造和移动赋值
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // 构造空Tensor（指定设备）加上explicit防止隐式调用不存在的构造函数产生错误
    explicit Tensor(DeviceType device = DeviceType::CPU) : device_(device) {}

    // 从cv::Mat构造（深拷贝+自动转CHW+归一化）
    explicit Tensor(const cv::Mat& mat);

    // 根据形状和数据类型构造
    Tensor(const std::vector<int>& shape, DataType dtype, DeviceType device);

    ~Tensor() { 
#ifdef DBG_Tensor
        printf("~Tensor() ptr=%p\n", data_ptr_);
#endif
        release(); }

    // 设备间传输
    //类方法后面加上const表示函数内部不能修改Tensor的成员
    Tensor cputogpu() const;
    Tensor gputocpu() const;

    // 克隆
    Tensor clone() const;

    // 转换为cv::Mat（反归一化+CHW转HWC）
    cv::Mat toMat(float scale = 1.0f) const;

    // 访问器
    DeviceType device() const { return device_; }
    DataType dtype() const { return dtype_; }
    const std::vector<int>& shape() const { return shape_; }
    void* data() const  { return data_ptr_; }
    // void* data() const { return data_ptr_; }

    size_t bytes()const{
        size_t elements = 1;
        for (int dim : shape_) elements *= dim;
        return elements * (dtype_ == DataType::FLOAT32 ? sizeof(float) : sizeof(uint8_t));
    }

    /*打印机制
使用示例
// 创建3x2x2的浮点Tensor
tensor.print("demo_tensor");
// 输出：
// Tensor 'demo_tensor' shape: [3, 2, 2], device: CPU, dtype: float32
// [
//     [
//         [    1.0000,     2.0000, ...,     3.0000,     4.0000 ]
//         [    5.0000,     6.0000, ...,     7.0000,     8.0000 ]
//     ],
//     [
//         [    9.0000,    10.0000, ...,    11.0000,    12.0000 ]
//         ...
//     ]
// ]
// 打印第2个通道，第3行，前8个元素
tensor.print(2, 3, 8, "demo_tensor");
*/
    // 针对CHW排列的特殊打印
    //行打印
    void print(int channel, int row, int elements = 10, const std::string& name = "") const;
    //打印多元素
    void print(const std::string& name = "", size_t max_elements = 10) const;
    // // 打印某行的n-m列
    void print(int channel, int row, int start_col, int end_col, const std::string& name = "") const;
    void print_shape(const std::string& name = "") const;
private:
    void release() {
        if (data_ptr_) {
            if (device_ == DeviceType::CPU) {
                delete[] static_cast<uint8_t*>(data_ptr_);
            } else {
#ifdef USE_CUDA
                cudaFree(data_ptr_);
#endif
            }
            data_ptr_ = nullptr;
        }
    }

    template<typename T>
    void print_impl(const T* data, size_t max_elements) const;
    void allocate_memory(size_t bytes);
};

#endif