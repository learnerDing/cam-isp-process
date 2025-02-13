//This is isp_opencv_pipeline.cpp
#include "../include/Isp/isp_opencv_pipeline.h"

ISPOpenCVPipeline::ISPOpenCVPipeline(FrameQueue<cv::Mat>& frameQueue)
    : m_inputQueue(frameQueue) {}

ISPOpenCVPipeline::~ISPOpenCVPipeline() {}

// 新增输出队列管理实现
void ISPOpenCVPipeline::addInferOutputQueue(FrameQueue<cv::Mat>* queue) {
    m_inferQueues.push_back(queue);
}

void ISPOpenCVPipeline::addEncodeOutputQueue(FrameQueue<AVFrame>* queue) {
    m_encodeQueues.push_back(queue);
}
// 新增转换函数实现
std::shared_ptr<AVFrame> ISPOpenCVPipeline::convertToAVFrame(const cv::Mat& mat) {
    // 确保颜色空间正确（根据ISP输出格式调整）
    cv::Mat bgrMat;
    cv::cvtColor(mat, bgrMat, cv::COLOR_RGB2BGR); // 如果ISP输出是RGB则需要转换

    // 分配AVFrame
    std::shared_ptr<AVFrame> frame(av_frame_alloc(), [](AVFrame* f) { av_frame_free(&f); });
    frame->width = bgrMat.cols;
    frame->height = bgrMat.rows;
    frame->format = AV_PIX_FMT_YUV420P;
    
    // 分配缓冲区
    if (av_frame_get_buffer(frame.get(), 0) < 0) {
        throw std::runtime_error("无法分配AVFrame缓冲区");
    }

    // 执行转换
    Mat_bgr2AVframe_yuv(bgrMat, frame.get());
    
    return frame;
}

void ISPOpenCVPipeline::run() {
    while (true) {
        std::shared_ptr<cv::Mat> frame;
        if (!m_inputQueue.getFrame(frame)) {
            break; // 停止条件
        }

        // 执行 Debayer 处理
        cv::Mat Image = applyDebayer(*frame);

        // 预留其他模块处理
        applyAWB(Image);
        applyCCM(Image);
        applyGamma(Image);

        // 最终处理完成的图像在Image变量中
        cv::Mat processedImage = Image.clone();//深拷贝一张
        // 分发到推理队列
        if (!m_inferQueues.empty()) {
            auto inferFrame = std::make_shared<cv::Mat>(processedImage);
            for (auto* queue : m_inferQueues) {
                queue->addFrame(inferFrame);
            }
        }

        // 分发到编码队列
        if (!m_encodeQueues.empty()) {
            try {
                auto avFrame = convertToAVFrame(processedImage);//编码队列需要yuv420p的数据格式
                for (auto* queue : m_encodeQueues) {
                    queue->addFrame(avFrame);
                }
            } catch (const std::exception& e) {
                // 处理转换异常
                std::cerr << "AVFrame转换失败: " << e.what() << std::endl;
            }
        }

    }
}
/*第4版opencv算法优化，awb使用 Q16.16 定点数格式避免浮点转换开销，ccm 将浮点矩阵预转换为缩放整数，使用整数 SIMD 指令
Average Time (ms):
Debayer: 5.78725
AWB:     10.2197
CCM:     11.4764
Gamma:   2.67399
Total:   30.1578 达到要求
*/
cv::Mat ISPOpenCVPipeline::applyDebayer(const cv::Mat& bayerImage) {
    cv::Mat rgbImage;
    cv::cvtColor(bayerImage, rgbImage, cv::COLOR_BayerRG2RGB);
    return rgbImage;
}
//灰度世界算法
void ISPOpenCVPipeline::applyAWB(cv::Mat& image) {
    assert(image.channels() == 3);

    // 计算均值
    cv::Scalar mean = cv::mean(image);
    float K = (mean[0] + mean[1] + mean[2]) / 3.0f;
    float gains[3] = {K / mean[0], K / mean[1], K / mean[2]};

    // 转换为定点数 (Q8.8格式，根据实际增益范围调整)
    const int fixed_shift = 8;
    int16_t gainB = static_cast<int16_t>(gains[0] * (1 << fixed_shift));
    int16_t gainG = static_cast<int16_t>(gains[1] * (1 << fixed_shift));
    int16_t gainR = static_cast<int16_t>(gains[2] * (1 << fixed_shift));

    // NEON向量
    int16x8_t gainB_neon = vdupq_n_s16(gainB);
    int16x8_t gainG_neon = vdupq_n_s16(gainG);
    int16x8_t gainR_neon = vdupq_n_s16(gainR);

    #pragma omp parallel for
    for (int r = 0; r < image.rows; ++r) {
        uint8_t* ptr = image.ptr<uint8_t>(r);
        int c = 0;

        // 每次处理8像素 (24字节)
        const int neonStep = 8;
        for (; c <= (image.cols - neonStep)*3; c += neonStep*3) {
            // 加载RGB数据 [B0,G0,R0, B1,G1,R1,...]
            uint8x8x3_t pixels = vld3_u8(ptr + c);

            // 扩展为16位
            uint16x8_t B = vmovl_u8(pixels.val[0]);
            uint16x8_t G = vmovl_u8(pixels.val[1]);
            uint16x8_t R = vmovl_u8(pixels.val[2]);

            // 定点乘法
            B = vmulq_u16(B, vreinterpretq_u16_s16(gainB_neon));
            G = vmulq_u16(G, vreinterpretq_u16_s16(gainG_neon));
            R = vmulq_u16(R, vreinterpretq_u16_s16(gainR_neon));

            // 右移并压缩回8位
            B = vshrq_n_u16(B, fixed_shift);
            G = vshrq_n_u16(G, fixed_shift);
            R = vshrq_n_u16(R, fixed_shift);

            uint8x8_t resB = vqmovn_u16(B);
            uint8x8_t resG = vqmovn_u16(G);
            uint8x8_t resR = vqmovn_u16(R);

            // 存储结果
            vst3_u8(ptr + c, {resB, resG, resR});
        }

        // 处理剩余像素
        for (; c < image.cols*3; c += 3) {
            ptr[c]   = cv::saturate_cast<uchar>((ptr[c]   * gainB) >> fixed_shift);
            ptr[c+1] = cv::saturate_cast<uchar>((ptr[c+1] * gainG) >> fixed_shift);
            ptr[c+2] = cv::saturate_cast<uchar>((ptr[c+2] * gainR) >> fixed_shift);
        }
    }
}

void ISPOpenCVPipeline::applyCCM(cv::Mat& image) {
    const cv::Matx33f CCM(
        2.34403f,  -1.18042f, -0.296824f,
        0.00823594f, 1.44385f, -0.556513f,
        -0.0795542f, 0.0806464f, 0.909063f
    );

    // 转换为16位整数 (Q4.12格式)
    cv::Mat intImg;
    image.convertTo(intImg, CV_16SC3);

    // 矩阵参数转换为整数并广播到NEON寄存器
    const int scale = 12;
    int16_t ccm[9];
    for (int i = 0; i < 9; ++i)
        ccm[i] = static_cast<int16_t>(CCM.val[i] * (1 << scale));

    // 加载每个通道的系数到广播的向量
    const int16x4_t r_coeff_b = vdup_n_s16(ccm[0]);
    const int16x4_t r_coeff_g = vdup_n_s16(ccm[1]);
    const int16x4_t r_coeff_r = vdup_n_s16(ccm[2]);
    const int16x4_t g_coeff_b = vdup_n_s16(ccm[3]);
    const int16x4_t g_coeff_g = vdup_n_s16(ccm[4]);
    const int16x4_t g_coeff_r = vdup_n_s16(ccm[5]);
    const int16x4_t b_coeff_b = vdup_n_s16(ccm[6]);
    const int16x4_t b_coeff_g = vdup_n_s16(ccm[7]);
    const int16x4_t b_coeff_r = vdup_n_s16(ccm[8]);

    #pragma omp parallel for
    for (int r = 0; r < intImg.rows; ++r) {
        int16_t* ptr = intImg.ptr<int16_t>(r);
        int c = 0;

        // 每次处理4像素
        const int neonStep = 4;
        for (; c <= (intImg.cols - neonStep)*3; c += neonStep*3) {
            // 加载四个像素的BGR通道
            int16x8x3_t pixels = vld3q_s16(ptr + c);
            int16x8_t B = pixels.val[0];
            int16x8_t G = pixels.val[1];
            int16x8_t R = pixels.val[2];

            // 计算R通道
            int32x4_t newR_low = vmull_s16(vget_low_s16(B), r_coeff_b);
            newR_low = vmlal_s16(newR_low, vget_low_s16(G), r_coeff_g);
            newR_low = vmlal_s16(newR_low, vget_low_s16(R), r_coeff_r);
            int32x4_t newR_high = vmull_s16(vget_high_s16(B), r_coeff_b);
            newR_high = vmlal_s16(newR_high, vget_high_s16(G), r_coeff_g);
            newR_high = vmlal_s16(newR_high, vget_high_s16(R), r_coeff_r);

            // 计算G通道
            int32x4_t newG_low = vmull_s16(vget_low_s16(B), g_coeff_b);
            newG_low = vmlal_s16(newG_low, vget_low_s16(G), g_coeff_g);
            newG_low = vmlal_s16(newG_low, vget_low_s16(R), g_coeff_r);
            int32x4_t newG_high = vmull_s16(vget_high_s16(B), g_coeff_b);
            newG_high = vmlal_s16(newG_high, vget_high_s16(G), g_coeff_g);
            newG_high = vmlal_s16(newG_high, vget_high_s16(R), g_coeff_r);

            // 计算B通道
            int32x4_t newB_low = vmull_s16(vget_low_s16(B), b_coeff_b);
            newB_low = vmlal_s16(newB_low, vget_low_s16(G), b_coeff_g);
            newB_low = vmlal_s16(newB_low, vget_low_s16(R), b_coeff_r);
            int32x4_t newB_high = vmull_s16(vget_high_s16(B), b_coeff_b);
            newB_high = vmlal_s16(newB_high, vget_high_s16(G), b_coeff_g);
            newB_high = vmlal_s16(newB_high, vget_high_s16(R), b_coeff_r);

            // 右移和饱和
            newR_low = vshrq_n_s32(newR_low, scale);
            newR_high = vshrq_n_s32(newR_high, scale);
            newG_low = vshrq_n_s32(newG_low, scale);
            newG_high = vshrq_n_s32(newG_high, scale);
            newB_low = vshrq_n_s32(newB_low, scale);
            newB_high = vshrq_n_s32(newB_high, scale);

            int16x4_t resR_low = vqmovn_s32(newR_low);
            int16x4_t resR_high = vqmovn_s32(newR_high);
            int16x8_t resR = vcombine_s16(resR_low, resR_high);

            int16x4_t resG_low = vqmovn_s32(newG_low);
            int16x4_t resG_high = vqmovn_s32(newG_high);
            int16x8_t resG = vcombine_s16(resG_low, resG_high);

            int16x4_t resB_low = vqmovn_s32(newB_low);
            int16x4_t resB_high = vqmovn_s32(newB_high);
            int16x8_t resB = vcombine_s16(resB_low, resB_high);

            // 重新打包并存储结果
            pixels.val[0] = resB;
            pixels.val[1] = resG;
            pixels.val[2] = resR;
            vst3q_s16(ptr + c, pixels);
        }

        // 处理剩余像素
        for (; c < intImg.cols*3; c += 3) {
            int16_t B = ptr[c];
            int16_t G = ptr[c+1];
            int16_t R = ptr[c+2];

            ptr[c]   = cv::saturate_cast<short>((B*ccm[6] + G*ccm[7] + R*ccm[8]) >> scale);
            ptr[c+1] = cv::saturate_cast<short>((B*ccm[3] + G*ccm[4] + R*ccm[5]) >> scale);
            ptr[c+2] = cv::saturate_cast<short>((B*ccm[0] + G*ccm[1] + R*ccm[2]) >> scale);
        }
    }

    // 转换回8UC3
    intImg.convertTo(image, CV_8UC3);
}

void ISPOpenCVPipeline::applyGamma(cv::Mat& image) {
    const float gamma = 0.3f; // 默认 Gamma 值
    unsigned char LUT[256];
    for (int i = 0; i < 256; ++i) {
        LUT[i] = static_cast<uchar>(std::pow(i / 255.0f, gamma) * 255.0f);
    }

    if (image.channels() == 1) { // 单通道
        for (int r = 0; r < image.rows; ++r) {
            for (int c = 0; c < image.cols; ++c) {
                image.at<uchar>(r, c) = LUT[image.at<uchar>(r, c)];
            }
        }
    } else { // 多通道
        for (int r = 0; r < image.rows; ++r) {
            for (int c = 0; c < image.cols; ++c) {
                cv::Vec3b& pixel = image.at<cv::Vec3b>(r, c);
                pixel[0] = LUT[pixel[0]]; // Blue
                pixel[1] = LUT[pixel[1]]; // Green
                pixel[2] = LUT[pixel[2]]; // Red
            }
        }
    }
}