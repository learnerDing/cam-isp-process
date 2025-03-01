#include "FrameQueue.h"
#include "EncodeThread.h"

#include <iostream>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <ctime>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
}

int main(int argc, char** argv) {
    if (argc <= 4) {
        std::cerr << "Usage: " << argv[0] << " <output file> <codec name> <yuv file> <width> <height>\n";
        return 1;
    }

    const std::string filename = argv[1];
    const std::string codec_name = argv[2];
    const std::string yuv_filename = argv[3];
    const int width = std::stoi(argv[4]);
    const int height = std::stoi(argv[5]);

    // 查找编码器
    const AVCodec* codec = avcodec_find_encoder_by_name(codec_name.c_str());
    if (!codec) {
        std::cerr << "Codec '" << codec_name << "' not found\n";
        return 1;
    }

    // 分配编码器上下文
    AVCodecContext* c = avcodec_alloc_context3(codec);
    if (!c) {
        std::cerr << "Could not allocate video codec context\n";
        return 1;
    }

    // 设置编码器参数
    c->bit_rate = 400000;
    c->width = width;
    c->height = height;
    c->time_base = {1, 25};
    c->framerate = {25, 1};
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;

    if (codec->id == AV_CODEC_ID_H264) {
        av_opt_set(c->priv_data, "preset", "slow", 0);
    }

    // 打开编码器
    if (avcodec_open2(c, codec, nullptr) < 0) {
        std::cerr << "Could not open codec\n";
        avcodec_free_context(&c); // 释放编码器上下文
        return 1;
    }

    // 打开输出文件
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Could not open " << filename << "\n";
        avcodec_free_context(&c); // 释放编码器上下文
        return 1;
    }

    // 打开 YUV 输入文件
    std::ifstream yuv_file(yuv_filename, std::ios::binary);
    if (!yuv_file) {
        std::cerr << "Could not open " << yuv_filename << "\n";
        avcodec_free_context(&c); // 释放编码器上下文
        return 1;
    }

    // 分配帧
    AVFrame* frame = av_frame_alloc();
    if (!frame) {
        std::cerr << "Could not allocate video frame\n";
        avcodec_free_context(&c); // 释放编码器上下文
        return 1;
    }
    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;
    std::cout<<"frame->format = "<<frame->format<<"frame->height="<<frame->height<<"\n";
    // 分配帧缓冲区
    if (av_frame_get_buffer(frame, 8) < 0) {
        std::cerr << "Could not allocate the video frame data\n";
        av_frame_free(&frame); // 释放帧
        avcodec_free_context(&c); // 释放编码器上下文
        return 1;
    }

    // 创建帧队列和编码线程
    FrameQueue<AVFrame> frameQueue(10);
    EncodeThread encodeThread(frameQueue, c, outfile);

    encodeThread.start();

    // 编码 1500 帧的视频
    for (int i = 0; i < 1500; i++) {
        std::fflush(stdout);

        // 确保帧数据可写
        /*Ensure that the frame data is writable, avoiding data copy if possible.
        Do nothing if the frame is writable, allocate new buffers and copy the data if it is not*/
        if (av_frame_make_writable(frame) < 0) {
            std::cerr << "Could not make frame writable\n";
            av_frame_free(&frame); // 释放帧
            avcodec_free_context(&c); // 释放编码器上下文
            return 1;
        }

        // 从文件读取 YUV 数据
        yuv_file.read(reinterpret_cast<char*>(frame->data[0]), width * height);      // Y
        yuv_file.read(reinterpret_cast<char*>(frame->data[1]), width * height / 4);  // U
        yuv_file.read(reinterpret_cast<char*>(frame->data[2]), width * height / 4);  // V

        frame->pts = i;
        
        //frame深拷贝一张，然后包装成shared_ptr送入队列
        frameQueue.addFrame(std::shared_ptr<AVFrame>(av_frame_clone(frame)));
    }

    // 停止帧队列并等待编码线程完成
    frameQueue.stop();
    encodeThread.join();

    // 添加序列结束码
    const uint8_t endcode[] = {0, 0, 1, 0xb7}; // h264 文件结束码
    outfile.write(reinterpret_cast<const char*>(endcode), sizeof(endcode));

    // 释放资源
    av_frame_free(&frame);
    avcodec_free_context(&c);

    return 0;
}