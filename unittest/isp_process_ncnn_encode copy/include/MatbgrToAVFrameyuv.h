//This is MatbgrToAVFrameyuv.h
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <stdexcept>
extern "C" {
#include "libswscale/swscale.h"
#include "libavutil/avutil.h"
#include <libavutil/frame.h>
}
void Mat_bgr2AVframe_yuv(const cv::Mat& bgr_mat, AVFrame* yuv_avframe) {
    // 验证输入Mat格式
    if (bgr_mat.empty() || bgr_mat.type() != CV_8UC3 || !bgr_mat.isContinuous()) {
        throw std::invalid_argument("输入必须为非空连续BGR图像(CV_8UC3)");
    }

    const int width = bgr_mat.cols;
    const int height = bgr_mat.rows;

    // 验证AVFrame参数
    if (yuv_avframe->format != AV_PIX_FMT_YUV420P || 
        yuv_avframe->width != width || 
        yuv_avframe->height != height) {
        throw std::invalid_argument("AVFrame参数不匹配(YUV420P或尺寸不一致)");
    }

    // 获取或创建SWS上下文
    SwsContext* sws_ctx = sws_getCachedContext(nullptr,
        width, height, AV_PIX_FMT_BGR24,        // 输入格式
        width, height, AV_PIX_FMT_YUV420P,      // 输出格式
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr); // 算法选择

    if (!sws_ctx) {
        throw std::runtime_error("无法创建SWS转换上下文");
    }

    // 设置数据指针和步长
    const uint8_t* src_data[] = { bgr_mat.data };
    const int src_linesize[] = { static_cast<int>(bgr_mat.step) };

    uint8_t* dst_data[] = {
        yuv_avframe->data[0], 
        yuv_avframe->data[1], 
        yuv_avframe->data[2]
    };
    const int dst_linesize[] = {
        yuv_avframe->linesize[0],
        yuv_avframe->linesize[1],
        yuv_avframe->linesize[2]
    };

    // 执行转换
    sws_scale(sws_ctx,
        src_data, src_linesize,
        0, height,        // 起始行和总高度
        dst_data, dst_linesize
    );
}



// int main() {
//     // 生成测试图像 1920x1080 BGR图像
//     const int width = 1920;
//     const int height = 1080;
//     cv::Mat bgr_mat(height, width, CV_8UC3);
    
//     // 创建红色渐变图像 (BGR格式)
//     for (int y = 0; y < height; ++y) {
//         cv::Vec3b* row = bgr_mat.ptr<cv::Vec3b>(y);
//         for (int x = 0; x < width; ++x) {
//             row[x] = cv::Vec3b(0, 0, 255 * x / width); // 从黑到红渐变
//         }
//     }

//     // 创建目标AVFrame
//     AVFrame* yuv_frame = av_frame_alloc();
//     if (!yuv_frame) {
//         std::cerr << "无法分配AVFrame" << std::endl;
//         return -1;
//     }
    
//     yuv_frame->format = AV_PIX_FMT_YUV420P;
//     yuv_frame->width  = width;
//     yuv_frame->height  = height;
    
//     // 分配YUV缓冲区
//     if (av_frame_get_buffer(yuv_frame, 0) < 0) {
//         std::cerr << "无法分配帧缓冲区" << std::endl;
//         av_frame_free(&yuv_frame);
//         return -1;
//     }

//     // 预热转换（避免冷启动误差）
//     try {
//         Mat_bgr2AVframe_yuv(bgr_mat, yuv_frame);
//     } catch (const std::exception& e) {
//         std::cerr << "预热失败: " << e.what() << std::endl;
//         av_frame_free(&yuv_frame);
//         return -1;
//     }

//     // 执行性能测试（100次迭代）
//     const int iterations = 100;
//     auto start = std::chrono::high_resolution_clock::now();
    
//     for (int i = 0; i < iterations; ++i) {
//         try {
//             Mat_bgr2AVframe_yuv(bgr_mat, yuv_frame);
//         } catch (const std::exception& e) {
//             std::cerr << "转换失败: " << e.what() << std::endl;
//             break;
//         }
//     }
    
//     auto end = std::chrono::high_resolution_clock::now();
//     auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

//     // 输出结果
//     std::cout << "转换次数: " << iterations << "\n总耗时: " 
//               << duration.count() << " 毫秒\n平均耗时: "
//               << static_cast<double>(duration.count()) / iterations 
//               << " 毫秒/次" << std::endl;

//     // 清理资源
//     av_frame_free(&yuv_frame);
//     return 0;
// }
