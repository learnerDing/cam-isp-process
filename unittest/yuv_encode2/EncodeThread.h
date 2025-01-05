#ifndef ENCODETHREAD_H
#define ENCODETHREAD_H

#include "FrameQueue.h"
#include <thread>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>

class EncodeThread : public Thread {
public:
    EncodeThread(FrameQueue& frameQueue, AVCodecContext* enc_ctx, FILE* outfile);
    ~EncodeThread();

    void run() override;

private:
    FrameQueue& m_frameQueue;
    AVCodecContext* m_enc_ctx;
    FILE* m_outfile;
    AVPacket* m_pkt;

    void encode(AVFrame* frame);
};

#endif // ENCODETHREAD_H