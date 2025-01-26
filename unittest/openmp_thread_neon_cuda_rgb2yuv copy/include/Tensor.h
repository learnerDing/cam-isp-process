//This is Tensor.h
#include <vector>
#include <memory>
#include <mutex>
#include <iostream>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#define USE_CUDA
#include <cuda_runtime.h>
enum class DeviceType { CPU, GPU };
enum class DataType { UINT8, FLOAT32 };
/*
Tensor使用说明：Tensor数据类型用于将cv::Mat数据转化为能够支持cuda FP32运算的gpu数据
使用示例：
// 示例1：从OpenCV Mat创建Tensor并转换到GPU
cv::Mat img = cv::imread("test.jpg");
Tensor cpu_tensor(img);          // 转换为CHW格式的FP32 Tensor
Tensor gpu_tensor = cpu_tensor.to(DeviceType::GPU);

// 示例2：将GPU Tensor转换回CPU并保存为Mat
Tensor processed_gpu = ...;       // 假设经过GPU处理
Tensor processed_cpu = processed_gpu.to(DeviceType::CPU);
cv::Mat result = processed_cpu.toMat(255.0, 0.0);  // 反归一化到0-255
cv::imwrite("result.jpg", result);

// 示例3：创建空Tensor
Tensor empty_tensor;             // 默认创建在CPU的空Tensor

// 示例4：访问元数据
std::cout << "Tensor shape: C=" << cpu_tensor.channels() 
          << " H=" << cpu_tensor.height()
          << " W=" << cpu_tensor.width() << std::endl;

// 示例5：直接访问数据（CPU Tensor）
Tensor float_tensor(...);
float* data = float_tensor.data<float>();
data[0] = 1.0f;  // 直接修改第一个元素

//实例6：使用=赋值tensor变量，必须自定义移动赋值运算符才行，因为默认的移动赋值运算
会尝试复制不可复制的std::mutex锁导致失败
yuv_tensor  =  pipeline.process();
*/
class Tensor {
private:
    void* data_ptr_ = nullptr;
    size_t ref_count_ = 0;
    DeviceType device_ = DeviceType::CPU;
    DataType dtype_ = DataType::UINT8;
    std::vector<int> shape_;  // CHW 格式 [Channels, Height, Width]
    mutable std::mutex mutex_;

    // 内存分配（已适配 CHW 格式）
    void allocate_memory(size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (device_ == DeviceType::CPU) {
            data_ptr_ = new uint8_t[bytes];
        } else {
#ifdef USE_CUDA
            cudaMalloc(&data_ptr_, bytes);
#endif
        }
    }

    // 从 OpenCV Mat 深拷贝并转换为 CHW 格式（支持数据类型转换）
// 修改后的深拷贝函数（支持类型转换）
void deep_copy_from(const cv::Mat& mat, DataType target_dtype, double scale = 1.0, double shift = 0.0) {
    CV_Assert(mat.isContinuous());
    const int channels = mat.channels();
    const int height = mat.rows;
    const int width = mat.cols;

    // 设置目标类型（强制为FLOAT32）
    dtype_ = DataType::FLOAT32;  // 强制设为FP32
    const size_t elem_size = sizeof(float);
    const size_t bytes = channels * height * width * elem_size;
    
    // 分配内存（使用float指针）
    allocate_memory(bytes);
    float* dest = static_cast<float*>(data_ptr_);

    // 执行类型转换和HWC->CHW转换
    cv::Mat converted_mat;
    mat.convertTo(converted_mat, CV_32F, scale, shift); // 转换为fp32

    std::vector<cv::Mat> channels_mat;
    cv::split(converted_mat, channels_mat);

    // 逐通道拷贝（现在处理的是float数据）
    for (int c = 0; c < channels; ++c) {
        CV_Assert(channels_mat[c].isContinuous());
        const float* src = channels_mat[c].ptr<float>(0);
        
        // 每个通道的像素数量
        const int num_pixels = height * width;
        
        // 使用memcpy直接拷贝连续内存
        std::memcpy(dest + c * num_pixels, src, num_pixels * sizeof(float));
    }

    ref_count_ = 1;
}

public:

// 默认构造函数（创建空CPU Tensor）
    Tensor() = default;
    // 深拷贝构造函数（自动转换 HWC->CHW，支持数据类型转换）
explicit Tensor(const cv::Mat& mat, 
               double scale = 1.0,
               double shift = 0.0) {
    CV_Assert(mat.depth() == CV_8U || mat.depth() == CV_32F);
    shape_ = {mat.channels(), mat.rows, mat.cols};
    dtype_ = DataType::FLOAT32;  // 强制设为FP32
    deep_copy_from(mat, dtype_,scale, shift);
}
    // 移动构造函数（浅拷贝）
    Tensor(Tensor&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.mutex_);
        data_ptr_ = other.data_ptr_;
        ref_count_ = other.ref_count_;
        device_ = other.device_;
        dtype_ = other.dtype_;
        shape_ = std::move(other.shape_);
        other.data_ptr_ = nullptr;
        other.ref_count_ = 0;
    }
    // 移动赋值运算符（浅拷贝）
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            std::lock_guard<std::mutex> lock(mutex_);
            release(); // 释放当前资源
            data_ptr_ = other.data_ptr_;
            ref_count_ = other.ref_count_;
            device_ = other.device_;
            dtype_ = other.dtype_;
            shape_ = std::move(other.shape_);
            other.data_ptr_ = nullptr;
            other.ref_count_ = 0;
        }
        return *this;
    }
    ~Tensor() { release(); }

    // 引用计数管理
    void add_ref() { 
        std::lock_guard<std::mutex> lock(mutex_);
        ++ref_count_; 
    }
    
    void release() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (--ref_count_ == 0) {
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

    // 设备间数据传输（保持 CHW 格式）
    Tensor to(DeviceType target) const {
        if (device_ == target) return *this;
        
        Tensor new_tensor;
        new_tensor.device_ = target;
        new_tensor.dtype_ = dtype_;
        new_tensor.shape_ = shape_;
        
        const size_t bytes = this->bytes();
        new_tensor.allocate_memory(bytes);
        
#ifdef USE_CUDA
        //cpu Tensor转化为gpu Tensor
        if (target == DeviceType::GPU) {
            cudaMemcpy(new_tensor.data_ptr_, data_ptr_, bytes, cudaMemcpyHostToDevice);
        } 
        else {//gpu Tensor转化为cpu Tensor
            cudaMemcpy(new_tensor.data_ptr_, data_ptr_, bytes, cudaMemcpyDeviceToHost);
        }
#endif
        new_tensor.ref_count_ = 1;
        return new_tensor;
    }

    // Tensor转换为OpenCV Mat（自动转换 CHW->HWC）
    cv::Mat toMat(double scale = 1.0, double shift = 0.0) const {
        Tensor cpu_tensor = this->to(DeviceType::CPU);
        const int C = channels();
        const int H = height();
        const int W = width();

        std::vector<cv::Mat> channels_mat;
        if (dtype_ == DataType::UINT8) {
            uint8_t* data = cpu_tensor.data<uint8_t>();
            for (int c = 0; c < C; ++c) {
                channels_mat.emplace_back(H, W, CV_8UC1, data + c * H * W);
            }
        } else {
            float* data = cpu_tensor.data<float>();
            for (int c = 0; c < C; ++c) {
                channels_mat.emplace_back(H, W, CV_32FC1, data + c * H * W);
            }
        }

        cv::Mat merged;
        cv::merge(channels_mat, merged);

        // 自动转换到目标类型
        if (dtype_ == DataType::FLOAT32) {
            cv::Mat converted;
            merged.convertTo(converted, CV_8U, scale, shift);
            return converted;
        }
        return merged;
    }

    // 数据访问接口
    template<typename T>
    T* data() { 
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<T*>(data_ptr_); 
    }

    // 形状信息
    const std::vector<int>& shape() const { return shape_; }
    int channels() const { return shape_[0]; }
    int height() const { return shape_[1]; }
    int width() const { return shape_[2]; }

    DeviceType device() const { return device_; }
    DataType dtype() const { return dtype_; }

    // 调试输出
    void print_element(size_t index) const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (device_ == DeviceType::GPU) {
            Tensor cpu_tensor = this->to(DeviceType::CPU);
            cpu_tensor.print_element(index);
            return;
        }
        
        switch(dtype_) {
            case DataType::UINT8: 
                std::cout << static_cast<uint8_t*>(data_ptr_)[index] << std::endl;
                break;
            case DataType::FLOAT32:
                std::cout << static_cast<float*>(data_ptr_)[index] << std::endl;
                break;
        }
    }

private:
    // 计算总字节数（基于 CHW 格式）
    size_t bytes() const {
        return shape_[0] * shape_[1] * shape_[2] * 
               (dtype_ == DataType::UINT8 ? sizeof(uint8_t) : sizeof(float));
    }
};