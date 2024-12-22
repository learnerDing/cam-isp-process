//tensor.h抽象张量数据结构 可以理解为kernel和kernel之间的数据流
#pragma once
#include <vector>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <iostream>
//#include <cuda_fp16.h>
#include "string_utils.h"
#include "macro.h"
enum Device
{
    CPU_PINNED,
    CPU,
    GPU
};

enum  DataType
{
    FP32,
    FP16,
    INT8,
    UINT8,//RGB采用这种数据类型
    INT32,
    BOOL,
    BYTES,
    UNSUPPORTED
};

template<typename T>
DataType getTensorType()
{
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return FP32;
    }
    // else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
    //     return FP16;
    // }
    else if (std::is_same<T, int>::value || std::is_same<T, const int>::value) {
        return INT32;
    }
    else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return INT8;
    }
    else if (std::is_same<T, uint8_t>::value || std::is_same<T, const uint8_t>::value) {
        return UINT8;
    }
    else if (std::is_same<T, bool>::value || std::is_same<T, const bool>::value) {
        return BOOL;
    }
    else if (std::is_same<T, char>::value || std::is_same<T, const char>::value) {
        return BYTES;
    }
    else {
        return UNSUPPORTED;
    }
}
template<typename T>
class TensorWrapper;

//Tensor:张量的基础类，不存储具体信息，只描述张量的位置，类型，形状
struct Tensor {
    Device              location;
    DataType            dtype;
    std::vector<int>    shape;

    Tensor() = default;
    //重写构造函数
    Tensor(const Device location_, 
            const DataType dtype_,
            const std::vector<int> shape_):
            location(location_),
            dtype(dtype_),
            shape(shape_){}
    //定义虚函数，申明这个方法可能在子类被重写
    virtual int size() const {
        if (shape.size() == 0) {
            // TODO: add an reminder info
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
    }

    template<typename T>
    TensorWrapper<T>* as(){//as方法，用来强转类型
        return static_cast<TensorWrapper<T>*>(this);//Tensor类型强转为TensorWrapper类型
    }

    std::string DeviceString() const
    {
        static const std::unordered_map<Device, std::string> devicetring{//定义一个unordered_map配对Device和string
            {CPU, "CPU"}, {CPU_PINNED, "CPU_PINNED"}, {GPU, "GPU"}};
        return devicetring.at(location);
    }

    virtual std::string toString() const
    {
        std::string device_str = DeviceString();

        static const std::unordered_map<DataType, std::string> type_to_string{
            {INT8, "INT8"},
            {INT32,"INT32"},
            {FP16, "FP16"},
            {FP32, "FP32"},

        };
        return fmtstr("Tensor[where=%s, type=%s, shape=%s]",
                    device_str.c_str(),
                    type_to_string.at(dtype).c_str(),
                    vec2str(shape).c_str());
    }  
};

template<typename T>
class TensorWrapper: public Tensor {
public:
    T* data;
    // cant declare shape's type to std::vector<int>&, because we usually pass a tmp var, which cant be non-const refer
    //https://blog.csdn.net/lisemi/article/details/103928596
    TensorWrapper(Device location, DataType dtype, std::vector<int> shape)://TensorWrapper第一种构造方式
    	Tensor(location, dtype, shape){}
        //数据格式Tensorwrapper->shape[3,2464,3264]  2464行，3264列 CHW排列
    TensorWrapper(Device location, DataType dtype, std::vector<int> shape, T* data)://TensorWrapper第二种构造方式
    	Tensor(location, dtype, shape),
    	data(data){
            DataType in_dtype = getTensorType<T>();//检测模板T的数据类型(eg.int,float,half)
            LLM_CHECK_WITH_INFO(in_dtype == dtype, "when build TensorWrapper, the passed in data type should be same as dtype in params");
        }
     // 深拷贝构造函数
    TensorWrapper(const TensorWrapper<T>* copyFromObj) : Tensor(*copyFromObj), data(nullptr) {
        if (copyFromObj != nullptr && copyFromObj->data != nullptr) {
            size_t num_elements = copyFromObj->size();
            data = new T[num_elements]; // 分配新的内存
            std::copy(copyFromObj->data, copyFromObj->data + num_elements, data); // 复制数据
        }
    }


    // Destructor to free the allocated data
    ~TensorWrapper() {
        delete[] data; // Delete the allocated memory
    }
    // friend bool operator==(Tensor& t1, Tensor& t2);
    //TensorWarp:size()方法：计算TensorWarp大小
    virtual int size() const {
        if (data == nullptr || shape.size() == 0) {
            // TODO: add an reminder info
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), (int)1, std::multiplies<int>());
    }
    //TensorWarp:getVal方法：返回tensor[id]的数据,只能检测位于内存的数据
    inline T getVal(int id) const {
        //TODO: need some boundry and device check
        LLM_CHECK(location == CPU);
        return data[id];
    } // only available on CPU by []

    inline T getVal() const
    {
        // TODO: add type check, this is very important, because we often naturally access GPU data, which is wrong
        // for example, I am in transpose kernel to use layer_id->getVal<int>(), which is wrong
        LLM_CHECK(location == CPU);
        return getVal(0);
    }
    inline T* getVal(int begin,int end) const {
        //TODO: need some boundry and device check
        T* temp = NULL;
        int j=begin;
        for(int i=0;i<=end-begin;i++)
        {   
            temp[i]=data[j++];
        }
        return temp;
    } //返回数组指针

//TensorWarp:getPtr 获取数据指针
    inline T* getPtr() const {
        //TODO: need some boundry check
        return (T*)data;
    }
//TensorWarp:getPtrByOffset 获取指针偏移之后的数据
    inline T* getPtrByOffset(int offset) const {
        //TODO: need some boundry check
        return (T*)data + offset;
    }
    // for debug
    virtual std::string toString() const
    {
        std::string device_str = DeviceString();

        static const std::unordered_map<DataType, std::string> type_to_string{
            {INT8, "INT8"},
            {FP16, "FP16"},
            {FP32, "FP32"},

        };
        return fmtstr("Tensor[where=%s, type=%s, shape=%s, data=%p]",
                    device_str.c_str(),
                    type_to_string.at(dtype).c_str(),
                    vec2str(shape).c_str(),
                    data);
    }    
};



//I cant check if the data pointer in TensorWrapper is nullptr, because the val in tensormap is tensor*
//so I must check the data pointer using LLM_CHECK_WITH_INFO before insert into tensormap.
#if 0
struct TensorMap {
    std::unordered_map<std::string, Tensor*> tensor_map_;//采用一个无序字典来存储tensor*，使用指针来减少拷贝和栈空间占用
    //tensor*是基类，才可以指向不同类型的TensorWrapper<T>
    TensorMap() = default;
    /*
    这里定义了一个名为TensorMap的构造函数，
    它接收一个std::initializer_list<std::pair<std::string, Tensor*>>类型的参数tensor_map。
    这个参数是一个初始化列表，其中包含多个键值对，键是字符串类型，值是指向Tensor对象的指针。*/

    TensorMap(std::initializer_list<std::pair<std::string, Tensor*>> tensor_map){
        for (auto& pair : tensor_map) {
            if (isValid(pair.second)) {//键值对的第二个值（指针不为空）
                insert(pair.first, pair.second);//将键值对pair<string,Tensor*>插入我们定义的unpdered_map里面
            }
            else {
                // std::cout << "this is not a valid tensor, skip to insert into tensormap" << std::endl;
                LLM_CHECK_WITH_INFO(isValid(pair.second),fmtstr("%s is not a valid tensor, skipping insert into TensorMap", pair.first.c_str()));
            }
        }
    }

    TensorMap(const std::unordered_map<std::string, Tensor*>& tensor_map) {
        // C++ 11 traverse
        // for (auto& kv : tensor_map) {
        // C++ 98 traverse
        for(auto it = tensor_map_.begin(); it != tensor_map_.end(); it++) {
            // if (isValid(kv.second)) {
            //     insert(kv.first, kv.second);
            // }
            if (isValid(it->second)) {
                insert(it->first, it->second);
            }
            else {
                // TODO: add a reminder info
            }
        }        
    };

    ~TensorMap(){
        tensor_map_.clear();
    }

    inline size_t size() const
    {
        return tensor_map_.size();
    }

    inline bool isExist(const std::string& key) const
    {
        return tensor_map_.find(key) != tensor_map_.end();
    }
    //isValid:检查TensorWarpper是否合规
    inline bool isValid(const Tensor* tensor)
    {
        return tensor->size() > 0;
    }
    // 增
    inline void insert(const std::string& key, Tensor* value)
    {
        // TODO: add a check to check key is unique and value is valid
        // tensor_map_.insert({key, value});
        tensor_map_[key] = value;
    }

    inline void insert(std::pair<std::string, Tensor*> p)
    {
        tensor_map_.insert(p);
    }
    //删

    //改

    //查
    inline Tensor* at(const std::string& key)//at方法，搜索Key值来查询是否存在
    {
         // TODO: add a check to check key is existed
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);
        
    }

    inline Tensor* operator[](const std::string& key)
    {
        LLM_CHECK_WITH_INFO(isExist(key), fmtstr("Cannot find a tensor of name %s in the tensor map    (keys: %s)",
                                  key.c_str(),
                                  vec2str(keys()).c_str()));
        return tensor_map_.at(key);

    }
   
    std::vector<std::string> keys() const
    {
        std::vector<std::string> key_names;
        for (auto& kv : tensor_map_) {
            key_names.push_back(kv.first);
        }
        return key_names;
    }
    // 打印出tensormap中的所有key
    std::string toString()
    {
        std::stringstream ss;
        ss << "{";
        std::vector<std::string> key_names = keys();
        for (size_t i = 0; i < tensor_map_.size(); ++i) {
            ss << key_names[i] << ": " << at(key_names[i])->toString();
            if (i < tensor_map_.size() - 1) {
                ss << ", ";
            }
        }
        ss << "}";
        return ss.str();
    }
};
#endif