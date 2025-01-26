#include "../include/Tensor.h"
Tensor::Tensor(Tensor&& other) noexcept 
    : data_ptr_(other.data_ptr_), device_(other.device_),
      dtype_(other.dtype_), shape_(std::move(other.shape_)) {
    other.data_ptr_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
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
Tensor::Tensor(const cv::Mat& mat) 
    : device_(DeviceType::CPU), dtype_(DataType::FLOAT32) {
    int channels = mat.channels();
    int height = mat.rows;
    int width = mat.cols;
    shape_ = {channels, height, width};

    cv::Mat mat_float;
    mat.convertTo(mat_float, CV_32F, 1.0/255.0);//uint_8转fp32
    if (!mat_float.isContinuous()) mat_float = mat_float.clone();

    size_t size = channels * height * width * sizeof(float);
    allocate_memory(size);//给当前Tensor分配cpu空间

    // HWC to CHW转换  此处必须深拷贝
    float* dst = static_cast<float*>(data_ptr_);
    const float* src = mat_float.ptr<float>();
    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                dst[c * height * width + h * width + w] = 
                    src[h * width * channels + w * channels + c];
            }
        }
    }
}
Tensor Tensor::to(DeviceType target) const {
    if (device_ == target) return clone();

    Tensor result(target);
    result.dtype_ = dtype_;
    result.shape_ = shape_;
    result.allocate_memory(bytes());

#ifdef USE_CUDA
    if (device_ == DeviceType::CPU && target == DeviceType::GPU) {
        cudaMemcpy(result.data_ptr_, data_ptr_, bytes(), cudaMemcpyHostToDevice);
    } else {
        cudaMemcpy(result.data_ptr_, data_ptr_, bytes(), cudaMemcpyDeviceToHost);
    }
#endif
    return result;
}

Tensor Tensor::cputogpu() const {
    assert(device_ == DeviceType::CPU);
    return to(DeviceType::GPU);
}

Tensor Tensor::gputocpu() const {
    assert(device_ == DeviceType::GPU);
    return to(DeviceType::CPU);
}
Tensor Tensor::clone() const {
    Tensor new_tensor;
    new_tensor.device_ = device_;
    new_tensor.dtype_ = dtype_;
    new_tensor.shape_ = shape_;
    
    const size_t data_size = bytes();
    new_tensor.allocate_memory(data_size);

    if (device_ == DeviceType::CPU) {
        // CPU内存拷贝
        memcpy(new_tensor.data_ptr_, data_ptr_, data_size);
    } else {
#ifdef USE_CUDA
        // GPU内存拷贝（Device-to-Device）
        cudaMemcpy(new_tensor.data_ptr_, data_ptr_, data_size, cudaMemcpyDeviceToDevice);
#endif
    }
    return new_tensor;
}
cv::Mat Tensor::toMat(float scale) const {
    // 前置检查
    assert(device_ == DeviceType::CPU && "toMat() requires CPU tensor");//设备检查：强制要求Tensor必须在CPU上
    assert(dtype_ == DataType::FLOAT32 && "Only FLOAT32 supported");
    assert(shape_.size() == 3 && "Need CHW shape");

    const int channels = shape_[0];
    const int height = shape_[1];
    const int width = shape_[2];

    // 创建临时Mat（CHW -> HWC转换）
    cv::Mat hwc_mat(height, width, CV_32FC(channels));
    float* dst = hwc_mat.ptr<float>();
    const float* src = static_cast<const float*>(data_ptr_);

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channels; ++c) {
                dst[h * width * channels + w * channels + c] = 
                    src[c * height * width + h * width + w] * scale;
            }
        }
    }

    // 转换为uint8（反归一化）
    cv::Mat output_mat;
    hwc_mat.convertTo(output_mat, CV_8UC(channels), scale);
    return output_mat;
}

void Tensor::allocate_memory(size_t bytes) {
    // 前置检查：确保当前没有内存泄漏
    assert(data_ptr_ == nullptr && "Tensor memory already allocated");

    if (device_ == DeviceType::CPU) {
        // 分配 CPU 内存（自动对齐）
        data_ptr_ = new uint8_t[bytes];
    } else {
#ifdef USE_CUDA
        // 分配 GPU 内存
        cudaError_t err = cudaMalloc(&data_ptr_, bytes);
        assert(err == cudaSuccess && "CUDA memory allocation failed");
#else
        assert(false && "CUDA support not enabled");
#endif
    }
}