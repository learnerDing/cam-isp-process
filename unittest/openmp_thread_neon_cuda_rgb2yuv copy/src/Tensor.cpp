//This is Tensor.cpp
#include "../include/Tensor.h"
#include <iomanip>
#include <iostream>
Tensor::Tensor(Tensor&& other) noexcept 
    : data_ptr_(other.data_ptr_), device_(other.device_),
      dtype_(other.dtype_), shape_(std::move(other.shape_)) {
    #ifdef DBG_Tensor 
        printf("Tensor(Tensor&&) move %p->%p by 移动构造\n", 
               other.data_ptr_, data_ptr_);
    #endif
    other.data_ptr_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    #ifdef DBG_Tensor 
        printf("Tensor(Tensor&&) move %p->%p by 移动赋值\n", 
               other.data_ptr_, data_ptr_);
    #endif
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
    #ifdef DBG_Tensor
        printf("Tensor create from Mat, shape=[ch:%d,h:%d,w:%d]\n",
               mat.channels(), mat.rows, mat.cols);
    #endif
    int channels = mat.channels();
    int height = height_ = mat.rows;
    int width = width_ = mat.cols;
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
//通过形状进行构造的构造函数
Tensor::Tensor(const std::vector<int>& shape, DataType dtype, DeviceType device)
    : shape_(shape), dtype_(dtype), device_(device) 
{
    // 检查形状有效性
    if (shape.empty()) {
        throw std::invalid_argument("Tensor shape cannot be empty");
    }
    for (int dim : shape) {
        if (dim <= 0) {
            throw std::invalid_argument("Tensor dimension must be positive");
        }
    }

    // 分配内存
    allocate_memory(bytes());
}
Tensor Tensor::to(DeviceType target) const {
    if (device_ == target) return clone();

    Tensor result(target);
    result.dtype_ = dtype_;
    result.shape_ = shape_;
    result.allocate_memory(bytes());

#ifdef USE_CUDA
    if (device_ == DeviceType::CPU && target == DeviceType::GPU) {//从cpu拷贝到gpu
        cudaMemcpy(result.data_ptr_, data_ptr_, bytes(), cudaMemcpyHostToDevice);
    } else {    //从gpu拷贝到cpu
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
// 新增YUV420P处理分支 检查形状是否是yuv420p排列
    if (shape_.size() == 3 && shape_[0] == 1 && 
    shape_[1] == 1 && 
    shape_[2] == (height_*width_ + 2*(height_/2)*(width_/2))) 
{
    printf("This is a yuv420p Tensor!\n");
    // 提取YUV分量
    const int y_size = height_ * width_;
    const int uv_size = (height_/2) * (width_/2);
    
    const float* y_plane = static_cast<const float*>(data_ptr_);
    const float* u_plane = y_plane + y_size;//u分量开始指针
    const float* v_plane = u_plane + uv_size;//v分量开始指针

    // 创建OpenCV YUV420P Mat（使用CV_8UC1类型）
    cv::Mat yuv420p_mat(height_ + height_/2, width_, CV_8UC1);
    uchar* dst = yuv420p_mat.ptr<uchar>();
    
    // 填充Y平面并应用缩放
    for (int i = 0; i < y_size; ++i) {
        dst[i] = static_cast<uchar>(y_plane[i] * scale);
    }
    
    // 填充U和V平面并应用缩放
    uchar* u_dst = dst + y_size;
    uchar* v_dst = u_dst + uv_size;
    for (int i = 0; i < uv_size; ++i) {
        u_dst[i] = static_cast<uchar>(u_plane[i] * scale);
        v_dst[i] = static_cast<uchar>(v_plane[i] * scale);
    }

    // 转换为BGR
    cv::Mat BGR_mat;
    cv::cvtColor(yuv420p_mat, BGR_mat, cv::COLOR_YUV2BGR_I420);
    // cv::namedWindow("BGR Image", cv::WINDOW_NORMAL);
    // cv::imshow("BGR Image",BGR_mat);
    return BGR_mat; // 已经是CV_8UC3类型，无需额外转换
} 
    else
    {//常规数据排列方式bgr各占4个字节
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
}

void Tensor::allocate_memory(size_t bytes) {
    // 前置检查：确保当前没有内存泄漏
    assert(data_ptr_ == nullptr && "Tensor memory already allocated");
    #ifdef DBG_Tensor
        printf("Allocate %zu bytes on %s\n", 
               bytes, device_==DeviceType::CPU?"CPU":"GPU");
    #endif
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

//打印机制
void Tensor::print_shape(const std::string& name) const {
    std::cout << "Tensor";
    if (!name.empty()) std::cout << " '" << name << "'";
    std::cout << " shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << shape_[i];
    }
    std::cout << "], device: " << (device_ == DeviceType::CPU ? "CPU" : "GPU")
              << ", dtype: " << (dtype_ == DataType::FLOAT32 ? "float32" : "uint8")
              << std::endl;
}

void Tensor::print(const std::string& name, size_t max_elements) const {
    print_shape(name);

    // 处理设备
    Tensor cpu_tensor;
    const Tensor* print_tensor = this;
    if (device_ != DeviceType::CPU) {
        cpu_tensor = this->to(DeviceType::CPU);
        print_tensor = &cpu_tensor;
    }

    // 空Tensor检查
    if (!print_tensor->data_ptr_) {
        std::cout << "Tensor data is empty!" << std::endl;
        return;
    }

    // 根据数据类型调用实现
    if (dtype_ == DataType::FLOAT32) {
        print_impl<float>(static_cast<const float*>(print_tensor->data()), max_elements);
    } else {
        print_impl<uint8_t>(static_cast<const uint8_t*>(print_tensor->data()), max_elements);
    }
}
void Tensor::print(int channel, int row, int elements, const std::string& name) const {
    print_shape(name);  // 复用形状打印
    
    // 设备检查与数据转移
    Tensor cpu_tensor;
    const Tensor* print_tensor = this;
    if (device_ != DeviceType::CPU) {
        cpu_tensor = this->to(DeviceType::CPU);
        print_tensor = &cpu_tensor;
    }

    // 参数有效性验证
    const auto& s = shape_;
    assert(s.size() == 3 && "Need CHW shape for this print method");
    printf("channel = %d,s[0] = %d\n",channel,s[0]);
    assert(channel >= 0 && channel < s[0] && "Channel index out of range");
    assert(row >= 0 && row < s[1] && "Row index out of range");
    elements = std::min(elements, s[2]);  // 不能超过实际列数

    // 数据指针定位
    const size_t offset = channel * s[1] * s[2] + row * s[2];
    
    // 统一格式化输出
    std::cout << "Printing CHW tensor data [C=" << channel 
              << ", H=" << row << ", W=0~" << (elements-1) << "]:\n";
    
    if (dtype_ == DataType::FLOAT32) {
        const float* data = static_cast<const float*>(print_tensor->data()) + offset;
        for (int i = 0; i < elements; ++i) {
            std::cout << std::fixed << std::setprecision(4) 
                      << std::setw(8) << data[i] << " ";
        }
    } else {
        const uint8_t* data = static_cast<const uint8_t*>(print_tensor->data()) + offset;
        for (int i = 0; i < elements; ++i) {
            std::cout << std::setw(4) << static_cast<int>(data[i]) << " ";
        }
    }
    std::cout << "\n..." << (s[2] > elements ? " (truncated)" : "") 
              << " Total elements in this row: " << s[2] << "\n\n";
}

template<typename T>
void Tensor::print_impl(const T* data, size_t max_elements) const {
    size_t total_elements = shape_.empty() ? 0 : 1;
    for (int dim : shape_) total_elements *= dim;

    std::cout << "[" << std::endl;
    
    // 递归打印多维数据的辅助函数
    std::function<void(const T*, size_t, size_t, std::vector<size_t>)> print_recursive;
    print_recursive = [&](const T* ptr, size_t dim_idx, size_t indent, std::vector<size_t> indices) {
        std::string indent_str(indent * 4, ' ');
        
        if (dim_idx == shape_.size() - 1) { // 最后一维
            std::cout << indent_str << "[ ";
            size_t elements_to_print = std::min<size_t>(shape_[dim_idx], 8);
            bool truncated = false;
            
            for (size_t i = 0; i < shape_[dim_idx]; ++i) {
                if (i >= 4 && i < shape_[dim_idx] - 4) {
                    if (!truncated) {
                        std::cout << "..., ";
                        truncated = true;
                    }
                    continue;
                }
                
                if (std::is_same<T, float>::value) {
                    std::cout << std::fixed << std::setw(9) << std::setprecision(4) << ptr[i];
                } else {
                    std::cout << std::setw(4) << static_cast<int>(ptr[i]);
                }
                
                if (i < shape_[dim_idx] - 1) std::cout << ", ";
            }
            std::cout << " ]" << std::endl;
        } else { // 中间维度
            size_t elements_to_print = std::min<size_t>(shape_[dim_idx], 3);
            bool truncated = shape_[dim_idx] > 6;
            
            for (size_t i = 0; i < shape_[dim_idx]; ++i) {
                if (i >= 3 && i < shape_[dim_idx] - 3) {
                    if (!truncated) {
                        std::cout << indent_str << "... (shape[" << dim_idx << "] = " 
                                  << shape_[dim_idx] << " elements)" << std::endl;
                        truncated = true;
                    }
                    continue;
                }
                
                std::cout << indent_str << "[";
                if (dim_idx < shape_.size() - 1) std::cout << std::endl;
                
                std::vector<size_t> new_indices = indices;
                new_indices.push_back(i);
                print_recursive(ptr + i * shape_[dim_idx + 1], dim_idx + 1, indent + 1, new_indices);
                
                if (dim_idx < shape_.size() - 1) {
                    std::cout << indent_str << "]";
                    if (i < shape_[dim_idx] - 1) std::cout << ",";
                    std::cout << std::endl;
                }
            }
        }
    };
    
    print_recursive(data, 0, 0, {});
    std::cout << "]" << std::endl;
    
    // 显示统计信息
    if (total_elements > max_elements) {
        std::cout << "... (showing first " << max_elements 
                  << " of " << total_elements << " elements)" << std::endl;
    }
}