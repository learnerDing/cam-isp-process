#pragma once
#include "FrameQueue.h"
#include "Thread.h"
#include <fstream> // 添加这行
#include <chrono>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/error.h>
}
//template <typename T>//T=AVFrame
class EncodeThread : public Thread {
public:
    EncodeThread(FrameQueue<AVFrame>& frameQueue, const std::string& codecName, 
                 int width, int height, const std::string& outputBase);
    ~EncodeThread();

    void stopEncoding();

protected:
    void run() override;

private:
    void initCodec(const std::string& codecName);
    void openNewOutputFile();
    std::string generateOutputFilename() const;
    void encode(AVFrame* frame);

    FrameQueue<AVFrame>& m_frameQueue;
    AVCodecContext* m_codecCtx;
    AVPacket* m_pkt;
    std::ofstream m_outputFile;
    std::string m_outputBase;
    int m_width;
    int m_height;
    std::chrono::time_point<std::chrono::system_clock> m_lastSplitTime;
    bool m_running;
    int m_failCount;
};