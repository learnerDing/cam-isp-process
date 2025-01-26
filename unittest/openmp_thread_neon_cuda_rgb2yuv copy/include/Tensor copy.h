//This is Tensor.h
#include <vector>
#include <memory>
// #include <mutex>
#include <iostream>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
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
    DeviceType device_ = DeviceType::CPU;
    DataType dtype_ = DataType::FLOAT32; // ISP处理一般用FP32
    std::vector<int> shape_;  // CHW格式

public:
    // 删除拷贝构造和拷贝赋值（强制移动语义）
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // 移动构造
    Tensor(Tensor&& other) noexcept 
        : data_ptr_(other.data_ptr_), 
          device_(other.device_),
          dtype_(other.dtype_),
          shape_(std::move(other.shape_)) {
        other.data_ptr_ = nullptr;
    }

    // 移动赋值
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            release();
            data_ptr_ = other.data_ptr_;
            device_ = other.device_;
            dtype_ = other.dtype_;
            shape_ = std::move(other.shape_);
            other.data_ptr_ = nullptr;
        }
        return *this;
    }

    // 构造空Tensor
    explicit Tensor(DeviceType device = DeviceType::CPU) : device_(device) {}

    // 从cv::Mat构造（深拷贝+自动转CHW）
    explicit Tensor(const cv::Mat& mat, DeviceType device = DeviceType::CPU) {
        // ...原有深拷贝逻辑（需适配无锁）
        allocate_memory(bytes);
        convert_hwc_to_chw(mat); 
    }

    // 析构时直接释放资源
    ~Tensor() { release(); }

    // 设备间传输（深拷贝）
    Tensor to(DeviceType target) const {
        if (device_ == target) {
            return clone(); // 同设备也需要深拷贝（避免线程竞争）
        }
        Tensor new_tensor;
        new_tensor.device_ = target;
        new_tensor.dtype_ = dtype_;
        new_tensor.shape_ = shape_;
        new_tensor.allocate_memory(bytes());
#ifdef USE_CUDA
        if (device_ == DeviceType::CPU && target == DeviceType::GPU) {
            cudaMemcpy(new_tensor.data_ptr_, data_ptr_, bytes(), cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(new_tensor.data_ptr_, data_ptr_, bytes(), cudaMemcpyDeviceToHost);
        }
#endif
        return new_tensor;
    }

    // 显式克隆接口
    Tensor clone() const {
        Tensor new_tensor;
        new_tensor.device_ = device_;
        new_tensor.dtype_ = dtype_;
        new_tensor.shape_ = shape_;
        new_tensor.allocate_memory(bytes());
        if (device_ == DeviceType::CPU) {
            memcpy(new_tensor.data_ptr_, data_ptr_, bytes());
        } else {
#ifdef USE_CUDA
            cudaMemcpy(new_tensor.data_ptr_, data_ptr_, bytes(), cudaMemcpyDeviceToDevice);
#endif
        }
        return new_tensor;
    }

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

    void allocate_memory(size_t bytes) {
        if (device_ == DeviceType::CPU) {
            data_ptr_ = new uint8_t[bytes];
        } else {
#ifdef USE_CUDA
            cudaMalloc(&data_ptr_, bytes);
#endif
        }
    }
};