#ifndef _TENSOR_H
#define _TENSOR_H
#include <vector>
#include <cassert>
#include <opencv2/core.hpp>
#include <cuda_runtime.h>

enum class DeviceType { CPU, GPU };
enum class DataType { UINT8, FLOAT32 };
#define USE_CUDA
class Tensor {
private:
    void* data_ptr_ = nullptr;
    DeviceType device_ = DeviceType::CPU;
    DataType dtype_ = DataType::FLOAT32;
    std::vector<int> shape_;

public:
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

    ~Tensor() { release(); }

    // 设备间传输
    Tensor to(DeviceType target) const;
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
    void* data() { return data_ptr_; }
    const void* data() const { return data_ptr_; }

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
    size_t bytes() const {
        size_t elements = 1;
        for (int dim : shape_) elements *= dim;
        return elements * (dtype_ == DataType::FLOAT32 ? sizeof(float) : sizeof(uint8_t));
    }
    void allocate_memory(size_t bytes);
};

#endif