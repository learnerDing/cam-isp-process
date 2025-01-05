#include "EncodeThread.h"

EncodeThread::EncodeThread(FrameQueue& frameQueue, AVCodecContext* enc_ctx, FILE* outfile)
    : m_frameQueue(frameQueue), m_enc_ctx(enc_ctx), m_outfile(outfile) {
    m_pkt = av_packet_alloc();
    if (!m_pkt) {
        fprintf(stderr, "Could not allocate AVPacket\n");
        exit(1);
    }
}

EncodeThread::~EncodeThread() {
    av_packet_free(&m_pkt);
}

void EncodeThread::run() {
    AVFrame* frame = nullptr;
    while (m_frameQueue.getFrame(&frame)) {
        encode(frame);
        av_frame_free(&frame);
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
        fprintf(stderr, "Error sending a frame for encoding\n");
        exit(1);
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(m_enc_ctx, m_pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            fprintf(stderr, "Error during encoding\n");
            exit(1);
        }

        printf("Write packet %3"PRId64" (size=%5d)\n", m_pkt->pts, m_pkt->size);
        fwrite(m_pkt->data, 1, m_pkt->size, m_outfile);
        av_packet_unref(m_pkt);
    }
}