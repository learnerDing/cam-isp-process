#include "layer.h"
#include "net.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp> // 包含 OpenCV 的图像处理头文件
#include "yolov5.h"
#include <float.h>
#include <stdio.h>
#include <vector>

//#define YOLOV5_V60 1 //YOLOv5 v6.0
#define YOLOV5_V62 1 //YOLOv5 v6.2 导出 ONNX 模型方法 https://github.com/shaoshengsong/yolov5_62_export_ncnn

#if YOLOV5_V60 || YOLOV5_V62
#define MAX_STRIDE 64 // 设置最大步幅
#else
#define MAX_STRIDE 32
#endif

class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true; // 仅处理一个输入
    }

    // 前向传播
    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w; // 输入宽度
        int h = bottom_blob.h; // 输入高度
        int channels = bottom_blob.c; // 输入通道数

        int outw = w / 2; // 输出宽度
        int outh = h / 2; // 输出高度
        int outc = channels * 4; // 输出通道数

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator); // 创建输出 blob
        if (top_blob.empty())
            return -100; // 创建失败则返回错误码

        // 并行处理每个输出通道
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + (p / channels) / 2;
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr; // 将输入值赋给输出值

                    outptr += 1; // 移动输出指针
                    ptr += 2; // 移动输入指针
                }
                ptr += w; // 跳过当前行
            }
        }

        return 0; // 返回成功
    }
};

// 定义 YoloV5Focus 层的创建器
DEFINE_LAYER_CREATOR(YoloV5Focus)
//Object换做是在yolov5.h里面申明
// struct Object
// {
//     cv::Rect_<float> rect; // 目标的矩形框
//     int label; // 目标类别
//     float prob; // 置信度
// };

// 计算两个目标的交集面积
static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect; // 求交集矩形
    return inter.area(); // 返回交集面积
}

// 降序排列并原地排序
static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob; // 取中间点

    while (i <= j)
    {
        while (faceobjects[i].prob > p) // 从左侧找到小于 p 的点
            i++;

        while (faceobjects[j].prob < p) // 从右侧找到大于 p 的点
            j--;

        if (i <= j)
        {
            // 交换
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    // 并行化排序过程
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j); // 左侧排序
        }
        #pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right); // 右侧排序
        }
    }
}

// 对对象进行排序的封装
static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty()) // 如果没有对象直接返回
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1); // 对对象进行排序
}

// 非最大值抑制
static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear(); // 清空已选的对象

    const int n = faceobjects.size(); // 总个数

    std::vector<float> areas(n); // 保存每个目标的面积
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area(); // 计算面积
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1; // 默认保留
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label) // 如果不是无分类，且标签不相同
                continue;

            // 计算交并比
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area; // 计算并集面积
            // float IoU = inter_area / union_area
            // 如果交集/并集 > nms阈值，则不保留
            if (inter_area / union_area > nms_threshold)
                keep = 0; 
        }

        if (keep)
            picked.push_back(i); // 保留该目标索引
    }
}

// Sigmoid 函数
static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

// 生成建议框
static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h; // 特征图的高度

    int num_grid_x;
    int num_grid_y;
    // 根据图像宽高调整网格数
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5; // 类别数

    const int num_anchors = anchors.w / 2; // 锚框数量

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2]; // 取锚框宽
        const float anchor_h = anchors[q * 2 + 1]; // 取锚框高

        const ncnn::Mat feat = feat_blob.channel(q); // 获取特征通道

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j); // 指向特征点
                float box_confidence = sigmoid(featptr[4]); // 计算置信度
                if (box_confidence >= prob_threshold) // 大于阈值继续处理
                {
                    // 找到最大类别索引
                    int class_index = 0;
                    float class_score = -FLT_MAX;
                    for (int k = 0; k < num_class; k++)
                    {
                        float score = featptr[5 + k]; // 获取类别得分
                        if (score > class_score) // 更新最大得分类别
                        {
                            class_index = k;
                            class_score = score;
                        }
                    }
                    float confidence = box_confidence * sigmoid(class_score); // 计算综合置信度
                    if (confidence >= prob_threshold) // 大于阈值则生成目标
                    {
                        // 从特征图生成候选框
                        float dx = sigmoid(featptr[0]); // 计算 x 偏移
                        float dy = sigmoid(featptr[1]); // 计算 y 偏移
                        float dw = sigmoid(featptr[2]); // 计算宽偏移
                        float dh = sigmoid(featptr[3]); // 计算高偏移

                        // 通过偏移计算框的中心和大小
                        float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                        float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                        float pb_w = pow(dw * 2.f, 2) * anchor_w;
                        float pb_h = pow(dh * 2.f, 2) * anchor_h;

                        float x0 = pb_cx - pb_w * 0.5f; // 左上角 x
                        float y0 = pb_cy - pb_h * 0.5f; // 左上角 y
                        float x1 = pb_cx + pb_w * 0.5f; // 右下角 x
                        float y1 = pb_cy + pb_h * 0.5f; // 右下角 y

                        Object obj;
                        obj.rect.x = x0;
                        obj.rect.y = y0;
                        obj.rect.width = x1 - x0; // 宽度
                        obj.rect.height = y1 - y0; // 高度
                        obj.label = class_index; // 类别索引
                        obj.prob = confidence; // 置信度

                        objects.push_back(obj); // 添加到目标列表
                    }
                }
            }
        }
    }
}

// 进行对象检测
int detect_yolov5(ncnn::Net& net,const cv::Mat& bgr, std::vector<Object>& objects)
{
    //加载模型参数放到main.cpp里面去做
    // ncnn::Net yolov5;

    // yolov5.opt.use_vulkan_compute = true; // 启用 Vulkan 计算
    // // yolov5.opt.use_bf16_storage = true;

    // // 从 https://github.com/ultralytics/yolov5 获取的原始预训练模型
    // // ncnn 模型 https://github.com/nihui/ncnn-assets/tree/master/models
// #if YOLOV5_V62
//     if (yolov5.load_param("yolov5s_6.2.param")) // 加载模型参数文件
//         exit(-1);
//     if (yolov5.load_model("yolov5s_6.2.bin")) // 加载模型权重文件
//         exit(-1);
// #elif YOLOV5_V60
//     if (yolov5.load_param("yolov5s_6.0.param")) // 加载模型参数文件
//         exit(-1);
//     if (yolov5.load_model("yolov5s_6.0.bin")) // 加载模型权重文件
//         exit(-1);
// #else
//     yolov5.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator); // 注册自定义层

//     if (yolov5.load_param("yolov5s.param")) // 加载模型参数文件
//         exit(-1);
//     if (yolov5.load_model("yolov5s.bin")) // 加载模型权重文件
//         exit(-1);
// #endif

    // const int target_size = 640; // 目标尺寸
    const int target_size = 320;
    const float prob_threshold = 0.25f; // 置信度阈值
    const float nms_threshold = 0.45f; // NMS 阈值

    int img_w = bgr.cols; // 输入图像宽度
    int img_h = bgr.rows; // 输入图像高度

    // 归一化图像到最大步幅的倍数
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w; // 按宽度缩放
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h; // 按高度缩放
        h = target_size;
        w = w * scale;
    }

    // 将输入图像转换为 ncnn 格式
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    // 对目标矩形进行填充
    int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;//padding之后的图片放这里
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);//图像padding操作，避免图像经过卷积尺寸变小

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f}; // 归一化
    in_pad.substract_mean_normalize(0, norm_vals); // 进行归一化处理

    ncnn::Extractor ex = net.create_extractor();  // 使用传入的net参数

    ex.input("images", in_pad); // 输入数据

    std::vector<Object> proposals; // 保存候选框列表

    // 锚框设置，来自 yolov5/models/yolov5s.yaml

    // 步幅 8
    {
        ncnn::Mat out;
        ex.extract("output", out); // 获取输出结果

        ncnn::Mat anchors(6); // 定义锚框
        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;

        std::vector<Object> objects8; // 保存步幅 8 的对象
        generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8); // 生成候选框
        proposals.insert(proposals.end(), objects8.begin(), objects8.end()); // 插入到总候选框
    }

    // 步幅 16
    {
        ncnn::Mat out;
#if YOLOV5_V62
        ex.extract("353", out); // 提取输出
#elif YOLOV5_V60
        ex.extract("376", out); // 提取输出
#else
        ex.extract("781", out); // 提取输出
#endif

        ncnn::Mat anchors(6); // 定义锚框
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16; // 保存步幅 16 的对象
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16); // 生成候选框
        proposals.insert(proposals.end(), objects16.begin(), objects16.end()); // 插入到总候选框
    }

    // 步幅 32
    {
        ncnn::Mat out;
#if YOLOV5_V62
        ex.extract("367", out); // 提取输出
#elif YOLOV5_V60
        ex.extract("401", out); // 提取输出
#else
        ex.extract("801", out); // 提取输出
#endif

        ncnn::Mat anchors(6); // 定义锚框
        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;

        std::vector<Object> objects32; // 保存步幅 32 的对象
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32); // 生成候选框
        proposals.insert(proposals.end(), objects32.begin(), objects32.end()); // 插入到总候选框
    }

    // 根据置信度对所有候选框进行排序
    qsort_descent_inplace(proposals);

    // 通过 NMS 方法应用非最大值抑制
    std::vector<int> picked; // 保存筛选后的目标
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size(); // 获取总目标数

    objects.resize(count); // 调整最终目标数组大小
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]]; // 获取选中的目标

        // 将坐标调整为原始未填充的坐标
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // 裁剪
        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0; // 更新宽度
        objects[i].rect.height = y1 - y0; // 更新高度
    }

    return 0; // 返回成功
}

// 绘制检测到的目标
void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",//9
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",//19
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",//29
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",//37
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",//47
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",//57
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",//67
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",//77
        "hair drier", "toothbrush"//79
    };

    cv::Mat image = bgr.clone(); // 克隆输入图像

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height); // 打印检测结果

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0)); // 绘制矩形框

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100); // 标签和置信度

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine); // 计算文本大小

        int x = obj.rect.x; // 文本 x 坐标
        int y = obj.rect.y - label_size.height - baseLine; // 文本 y 坐标
        if (y < 0)
            y = 0; // 如果 y 超出图像范围则裁剪
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width; // 如果 x 超出图像范围则裁剪

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1); // 绘制背景框

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0)); // 在图像上绘制文本
    }

    cv::imshow("image", image); // 显示图像
    cv::waitKey(0); // 等待按键
}

// int main(int argc, char** argv)
// {
//     if (argc != 2) // 检查命令行参数数量
//     {
//         fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]); // 输出用法
//         return -1;
//     }

//     const char* imagepath = argv[1]; // 获取图像路径

//     cv::Mat m = cv::imread(imagepath, 1); // 读取图像
//     if (m.empty()) // 检查图像是否成功读取
//     {
//         fprintf(stderr, "cv::imread %s failed\n", imagepath); // 输出错误
//         return -1;
//     }

//     std::vector<Object> objects; // 保存检测到的目标
//     detect_yolov5(m, objects); // 进行目标检测

//     draw_objects(m, objects); // 绘制检测结果

//     return 0; // 返回成功
// }
