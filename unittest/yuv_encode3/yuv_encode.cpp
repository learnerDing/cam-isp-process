#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
#include <libavutil/error.h>
}

// 自定义删除器，用于释放 AVCodecContext
struct AVCodecContextDeleter {
    void operator()(AVCodecContext* ctx) const {
        avcodec_free_context(&ctx);
    }
};

// 自定义删除器，用于释放 AVPacket
struct AVPacketDeleter {
    void operator()(AVPacket* pkt) const {
        av_packet_free(&pkt);
    }
};

// 自定义删除器，用于释放 AVFrame
struct AVFrameDeleter {
    void operator()(AVFrame* frame) const {
        av_frame_free(&frame);
    }
};

static void encode(AVCodecContext* enc_ctx, AVFrame* frame, AVPacket* pkt, std::ofstream& outfile) {
    int ret;

    /* send the frame to the encoder */
    if (frame)
        std::cout << "Send frame " << frame->pts << std::endl;

    ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Error sending a frame for encoding: " << errbuf << std::endl;
        exit(1);
    }

    while (ret >= 0) {
        ret = avcodec_receive_packet(enc_ctx, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
            return;
        else if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
            std::cerr << "Error during encoding: " << errbuf << std::endl;
            exit(1);
        }

        std::cout << "Write packet " << pkt->pts << " (size=" << pkt->size << ")" << std::endl;
        outfile.write(reinterpret_cast<char*>(pkt->data), pkt->size);
        av_packet_unref(pkt);
    }
}

int main(int argc, char** argv) {
    if (argc <= 4) {
        std::cerr << "Usage: " << argv[0] << " <output file> <codec name> <yuv file> <width> <height>" << std::endl;
        exit(0);
    }

    std::string filename = argv[1];
    std::string codec_name = argv[2];
    std::string yuv_filename = argv[3];
    int width = std::stoi(argv[4]);
    int height = std::stoi(argv[5]);

    /* find the mpeg1video encoder */
    const AVCodec* codec = avcodec_find_encoder_by_name(codec_name.c_str());
    if (!codec) {
        std::cerr << "Codec '" << codec_name << "' not found" << std::endl;
        exit(1);
    }

    // 使用自定义删除器管理 AVCodecContext
    std::unique_ptr<AVCodecContext, AVCodecContextDeleter> c(avcodec_alloc_context3(codec));
    if (!c) {
        std::cerr << "Could not allocate video codec context" << std::endl;
        exit(1);
    }

    // 使用自定义删除器管理 AVPacket
    std::unique_ptr<AVPacket, AVPacketDeleter> pkt(av_packet_alloc());
    if (!pkt)
        exit(1);

    /* put sample parameters */
    c->bit_rate = 400000;
    /* resolution must be a multiple of two */
    c->width = width;
    c->height = height;
    /* frames per second */
    c->time_base = {1, 25};
    c->framerate = {25, 1};

    /* emit one intra frame every ten frames
     * check frame pict_type before passing frame
     * to encoder, if frame->pict_type is AV_PICTURE_TYPE_I
     * then gop_size is ignored and the output of encoder
     * will always be I frame irrespective to gop_size
     */
    c->gop_size = 10;
    c->max_b_frames = 1;
    c->pix_fmt = AV_PIX_FMT_YUV420P;

    if (codec->id == AV_CODEC_ID_H264)
        av_opt_set(c->priv_data, "preset", "slow", 0);

    /* open it */
    int ret = avcodec_open2(c.get(), codec, nullptr);
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Could not open codec: " << errbuf << std::endl;
        exit(1);
    }

    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Could not open " << filename << std::endl;
        exit(1);
    }

    std::ifstream yuv_file(yuv_filename, std::ios::binary);
    if (!yuv_file) {
        std::cerr << "Could not open " << yuv_filename << std::endl;
        exit(1);
    }

    // 使用自定义删除器管理 AVFrame
    std::unique_ptr<AVFrame, AVFrameDeleter> frame(av_frame_alloc());
    if (!frame) {
        std::cerr << "Could not allocate video frame" << std::endl;
        exit(1);
    }
    frame->format = c->pix_fmt;
    frame->width = c->width;
    frame->height = c->height;

    ret = av_frame_get_buffer(frame.get(), 8); // 经过测试，对齐方式8才ok
    if (ret < 0) {
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
        std::cerr << "Could not allocate the video frame data: " << errbuf << std::endl;
        exit(1);
    }

    /* encode 1 second of video */
    for (int i = 0; i < 1500; i++) {
        std::fflush(stdout);

        /* make sure the frame data is writable */
        ret = av_frame_make_writable(frame.get());
        if (ret < 0) {
            char errbuf[AV_ERROR_MAX_STRING_SIZE];
            av_make_error_string(errbuf, AV_ERROR_MAX_STRING_SIZE, ret);
            std::cerr << "Error making frame writable: " << errbuf << std::endl;
            exit(1);
        }

        /* read YUV data from file */
        yuv_file.read(reinterpret_cast<char*>(frame->data[0]), width * height); // Y
        yuv_file.read(reinterpret_cast<char*>(frame->data[1]), width * height / 4); // U
        yuv_file.read(reinterpret_cast<char*>(frame->data[2]), width * height / 4); // V

        frame->pts = i;

        /* encode the image */
        encode(c.get(), frame.get(), pkt.get(), outfile);
    }

    /* flush the encoder */
    for (int i = 0; i < 100; i++) {
        encode(c.get(), nullptr, pkt.get(), outfile);
    }

    /* add sequence end code to have a real MPEG file */
    uint8_t endcode[] = {0, 0, 1, 0xb7};
    outfile.write(reinterpret_cast<char*>(endcode), sizeof(endcode));
    outfile.close();
    yuv_file.close();

    return 0;
}