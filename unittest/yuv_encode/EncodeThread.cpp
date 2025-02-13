//This is EncodeThread.cpp
#include "EncodeThread.h"
#include <iostream>
#include <iomanip>
#include <sstream>

EncodeThread::EncodeThread(FrameQueue<AVFrame>& frameQueue, const std::string& codecName,
                           int width, int height, const std::string& outputBase)
    : m_frameQueue(frameQueue), m_outputBase(outputBase),
      m_width(width), m_height(height), m_running(true), m_failCount(0) {
    initCodec(codecName);
    m_pkt = av_packet_alloc();
    if (!m_pkt) throw std::runtime_error("Failed to allocate AVPacket");
    openNewOutputFile();
}

EncodeThread::~EncodeThread() {
    stopEncoding();
    avcodec_free_context(&m_codecCtx);
    av_packet_free(&m_pkt);
}
void EncodeThread::initCodec(const std::string& codecName) {
    const AVCodec* codec = avcodec_find_encoder_by_name(codecName.c_str());
    if (!codec) throw std::runtime_error("Codec not found");

    m_codecCtx = avcodec_alloc_context3(codec);
    if (!m_codecCtx) throw std::runtime_error("Failed to allocate codec context");

    // 配置编码参数
    m_codecCtx->bit_rate = 400000;
    m_codecCtx->width = m_width;
    m_codecCtx->height = m_height;
    m_codecCtx->time_base = {1, 25};
    m_codecCtx->framerate = {25, 1};
    m_codecCtx->gop_size = 10;
    m_codecCtx->max_b_frames = 1;
    m_codecCtx->pix_fmt = AV_PIX_FMT_YUV420P;

    if (codec->id == AV_CODEC_ID_H264)
        av_opt_set(m_codecCtx->priv_data, "preset", "slow", 0);

    if (avcodec_open2(m_codecCtx, codec, nullptr) < 0)
        throw std::runtime_error("Failed to open codec");
}

void EncodeThread::openNewOutputFile() {
    std::string filename = generateOutputFilename();
    m_outputFile.open(filename, std::ios::binary);
    if (!m_outputFile) throw std::runtime_error("Failed to open output file");
    m_lastSplitTime = std::chrono::system_clock::now();
}

std::string EncodeThread::generateOutputFilename() const {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&time);
    
    std::ostringstream oss;
    oss << m_outputBase << "_" 
        << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".h264";
    return oss.str();
}

void EncodeThread::run() {
    const int maxFails = 10;
    while (m_running && m_failCount < maxFails) {
        std::shared_ptr<AVFrame> frame;
        if (!m_frameQueue.getFrame(frame, 100)) { // 100ms 超时
            m_failCount++;
            continue;
        }
        
        m_failCount = 0; // 成功获取帧，重置失败计数器
        encode(frame.get());
        
        // 检查是否需要分割文件
        auto now = std::chrono::system_clock::now();
        if (now - m_lastSplitTime > std::chrono::minutes(10)) {
            m_outputFile.close();
            openNewOutputFile();
        }
    }
    
    // 刷新编码缓冲区
    encode(nullptr);
    
    // 写入结束码
    const uint8_t endcode[] = {0, 0, 1, 0xb7};
    m_outputFile.write(reinterpret_cast<const char*>(endcode), sizeof(endcode));
    m_outputFile.close();
}

void EncodeThread::encode(AVFrame* frame) {
    int ret = avcodec_send_frame(m_codecCtx, frame);
    if (ret < 0) throw std::runtime_error("Error sending frame");

    while (ret >= 0) {
        ret = avcodec_receive_packet(m_codecCtx, m_pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) throw std::runtime_error("Error during encoding");

        m_outputFile.write(reinterpret_cast<char*>(m_pkt->data), m_pkt->size);
        av_packet_unref(m_pkt);
    }
}

void EncodeThread::stopEncoding() {
    m_running = false;
}