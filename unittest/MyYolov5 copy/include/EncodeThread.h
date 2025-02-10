#ifndef ENCODETHREAD_H
#define ENCODETHREAD_H

#include "FrameQueue.h"
#include "Thread.h"
#include <fstream> // 添加这行
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/error.h>
}
//template <typename T>//T=AVFrame
class EncodeThread : public Thread {
public:
    EncodeThread(FrameQueue<AVFrame>& frameQueue, AVCodecContext* enc_ctx, std::ofstream& outfile);
    ~EncodeThread();

    void run() override;

private:
    FrameQueue<AVFrame>& m_frameQueue;
    AVCodecContext* m_enc_ctx;
    std::ofstream& m_outfile;
    AVPacket* m_pkt;

    void encode(AVFrame* frame);
};

#endif // ENCODETHREAD_H