//EncodeThread.cpp
#include "EncodeThread.h"

EncodeThread::EncodeThread(FrameQueue<AVFrame>& frameQueue, AVCodecContext* enc_ctx, std::ofstream& outfile)
    : m_frameQueue(frameQueue), m_enc_ctx(enc_ctx), m_outfile(outfile) {
    m_pkt = av_packet_alloc();
    if (!m_pkt) {
       throw std::runtime_error("Could not allocate AVPacket");
    }
}

EncodeThread::~EncodeThread() {
    av_packet_free(&m_pkt);
}

void EncodeThread::run() {
    AVFrame* frame = nullptr;
    //[](AVFrame* frame){if(frame)av_frame_free(&frame);})lambda表达式当智能指针删除器
    std::shared_ptr<AVFrame> frameptr(frame,[](AVFrame* frame)
    {
        if(frame)
        av_frame_free(&frame);// 使用 FFmpeg 的 API 释放 AVFrame
    }
    );
    while (m_frameQueue.getFrame(frameptr)) 
    {   
        frame = frameptr.get();
        encode(frame);
        //因为前面使用了shared_ptr<AVFrame>，在线程之外会自动析构，所以不需要再释放frame
        //av_frame_free(&frame);
    }

    // 刷新编码器
    encode(nullptr);
}

void EncodeThread::encode(AVFrame* frame) {
    int ret;

    /* send the frame to the encoder */
    if (frame)
        printf("Send frame %3"PRId64"\n", frame->pts);

    ret = avcodec_send_frame(m_enc_ctx, frame);
    if (ret < 0) {
                throw std::runtime_error("Error sending a frame for encoding"); 
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(m_enc_ctx, m_pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
          throw std::runtime_error("Error during encoding");
        }

        printf("Write packet %3"PRId64" (size=%5d)\n", m_pkt->pts, m_pkt->size);
        m_outfile.write(reinterpret_cast<const char*>(m_pkt->data), m_pkt->size);
        av_packet_unref(m_pkt);
    }
}