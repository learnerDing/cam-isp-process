// app.cpp
#include <iostream>      //for open printf
#include <fcntl.h>      //for open O_RDWR 文件控制定义
#include <cstring>      //for memset
#include <sys/ioctl.h>  //for ioctl
#include <linux/videodev2.h> //for v4l2

#include <sys/mman.h>
#include <unistd.h>     //Unix 标准函数定义
#include <cerrno>      //错误号定义

#define VIDEO_WIDTH  320  //采集图像的宽度
#define VIDEO_HEIGHT 240  //采集图像的高度

#define REQBUFS_COUNT 4     //缓存区个数
struct v4l2_requestbuffers reqbufs;  //定义缓冲区
struct cam_buf {
    void *start;
    size_t length;
};
struct cam_buf bufs[REQBUFS_COUNT]; //映射后指向的同一片帧缓冲区

//查看 摄像头设备的能力
int get_capability(int fd) {
    int ret = 0;
    struct v4l2_capability cap;

    memset(&cap, 0, sizeof(struct v4l2_capability)); /*SourceInsight跳转,可看到能力描述
            struct v4l2_capability {
            __u8	driver[16];   驱动名
            __u8	card[32];   设备名
            __u8	bus_info[32]; 总线信息
            __u32   version;  版本
            __u32	capabilities;  设备支持的操作
            __u32	device_caps;
            __u32	reserved[3];
        };
      */
    ret = ioctl(fd, VIDIOC_QUERYCAP, &cap);  //查看设备能力信息
    if (ret < 0) {
        std::cerr << "VIDIOC_QUERYCAP failed (" << ret << ")" << std::endl;
        return ret;
    }
    std::cout << "Driver Info: " << std::endl
              << "  Driver Name: " << cap.driver << std::endl
              << "  Card Name: " << cap.card << std::endl
              << "  Bus info: " << cap.bus_info << std::endl;
    std::cout << "Device capabilities: " << std::endl;
    if (cap.capabilities & V4L2_CAP_VIDEO_CAPTURE) { //支持视频捕获(截取一帧图像保存)
        std::cout << "  support video capture " << std::endl;
    }

    if (cap.capabilities & V4L2_CAP_STREAMING) { //支持视频流操作
        std::cout << "  support streaming i/o" << std::endl;
    }

    if (cap.capabilities & V4L2_CAP_READWRITE) { //支持读写
        std::cout << "  support read i/o" << std::endl;
    }
    return ret;
}

//查看 摄像头支持的视频格式
int get_suppurt_video_format(int fd) {
    int ret = 0;
    std::cout << "List device support video format:  " << std::endl;
    struct v4l2_fmtdesc fmtdesc;
    memset(&fmtdesc, 0, sizeof(fmtdesc));
    fmtdesc.index = 0;
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    while ((ret = ioctl(fd, VIDIOC_ENUM_FMT, &fmtdesc)) == 0) //枚举出支持的视频格式
    {
        fmtdesc.index++;
        std::cout << "  { pixelformat = '" 
                  << static_cast<char>(fmtdesc.pixelformat & 0xFF) << static_cast<char>((fmtdesc.pixelformat >> 8) & 0xFF)
                  << static_cast<char>((fmtdesc.pixelformat >> 16) & 0xFF) << static_cast<char>((fmtdesc.pixelformat >> 24) & 0xFF)
                  << "', description = '" << fmtdesc.description << "' }" << std::endl;
    }
    return ret;
}

//设置视频格式
int set_video_format(int fd) {
    int ret = 0;
    struct v4l2_format fmt;

    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = VIDEO_WIDTH;
    fmt.fmt.pix.height = VIDEO_HEIGHT;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_MJPEG; //设置为 MJPEG 格式
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    ret = ioctl(fd, VIDIOC_S_FMT, &fmt);
    if (ret < 0) {
        std::cerr << "VIDIOC_S_FMT failed (" << ret << ")" << std::endl;
        return ret;
    }

    // 获取视频格式
    ret = ioctl(fd, VIDIOC_G_FMT, &fmt);
    if (ret < 0) {
        std::cerr << "VIDIOC_G_FMT failed (" << ret << ")" << std::endl;
        return ret;
    }
    // Print Stream Format
    std::cout << "Stream Format Informations:" << std::endl;
    std::cout << " type: " << fmt.type << std::endl;
    std::cout << " width: " << fmt.fmt.pix.width << std::endl;
    std::cout << " height: " << fmt.fmt.pix.height << std::endl;
    char fmtstr[8];
    memset(fmtstr, 0, 8);
    memcpy(fmtstr, &fmt.fmt.pix.pixelformat, 4);
    std::cout << " pixelformat: " << fmtstr << std::endl;
    std::cout << " field: " << fmt.fmt.pix.field << std::endl;
    std::cout << " bytesperline: " << fmt.fmt.pix.bytesperline << std::endl;
    std::cout << " sizeimage: " << fmt.fmt.pix.sizeimage << std::endl;
    std::cout << " colorspace: " << fmt.fmt.pix.colorspace << std::endl;
    std::cout << " priv: " << fmt.fmt.pix.priv << std::endl;
    std::cout << " raw_data: " << fmt.fmt.raw_data << std::endl;
    return ret;
}

//申请帧缓冲区
int request_buf(int fd) {
    int ret = 0;
    int i;
    struct v4l2_buffer vbuf;

    memset(&reqbufs, 0, sizeof(struct v4l2_requestbuffers));
    reqbufs.count = REQBUFS_COUNT; //缓存区个数
    reqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    reqbufs.memory = V4L2_MEMORY_MMAP; //设置操作申请缓存的方式:映射 MMAP
    ret = ioctl(fd, VIDIOC_REQBUFS, &reqbufs); //向驱动申请缓存
    if (ret == -1) {
        std::cerr << "VIDIOC_REQBUFS fail  " << __FUNCTION__ << " " << __LINE__ << std::endl;
        return ret;
    }
    //循环映射并入队
    for (i = 0; i < reqbufs.count; i++) {
        //真正获取缓存的地址大小
        memset(&vbuf, 0, sizeof(struct v4l2_buffer));
        vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vbuf.memory = V4L2_MEMORY_MMAP;
        vbuf.index = i;
        ret = ioctl(fd, VIDIOC_QUERYBUF, &vbuf);
        if (ret == -1) {
            std::cerr << "VIDIOC_QUERYBUF fail  " << __FUNCTION__ << " " << __LINE__ << std::endl;
            return ret;
        }
        //映射缓存到用户空间
        bufs[i].length = vbuf.length;
        bufs[i].start = mmap(NULL, vbuf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, vbuf.m.offset);
        if (bufs[i].start == MAP_FAILED) {
            std::cerr << "mmap fail  " << __FUNCTION__ << " " << __LINE__ << std::endl;
            return ret;
        }
        //每次映射都会入队
        vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        vbuf.memory = V4L2_MEMORY_MMAP;
        ret = ioctl(fd, VIDIOC_QBUF, &vbuf);
        if (ret == -1) {
            std::cerr << "VIDIOC_QBUF err " << __FUNCTION__ << " " << __LINE__ << std::endl;
            return ret;
        }
    }
    return ret;
}

//启动采集
int start_camera(int fd) {
    int ret;

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ret = ioctl(fd, VIDIOC_STREAMON, &type); //ioctl控制摄像头开始采集
    if (ret == -1) {
        perror("start_camera");
        return -1;
    }
    std::cout << "camera->start: start capture" << std::endl;
    return 0;
}

//出队取一帧图像
int camera_dqbuf(int fd, void **buf, unsigned int *size, unsigned int *index) {
    int ret = 0;
    struct v4l2_buffer vbuf;
    vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vbuf.memory = V4L2_MEMORY_MMAP;
    ret = ioctl(fd, VIDIOC_DQBUF, &vbuf); //出队
    if (ret == -1) {
        perror("camera dqbuf ");
        return -1;
    }
    *buf = bufs[vbuf.index].start;
    *size = vbuf.bytesused;
    *index = vbuf.index;
    return ret;
}

//入队归还帧缓冲
int camera_eqbuf(int fd, unsigned int index) {
    int ret;
    struct v4l2_buffer vbuf;

    vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vbuf.memory = V4L2_MEMORY_MMAP;
    vbuf.index = index;
    ret = ioctl(fd, VIDIOC_QBUF, &vbuf); //入队
    if (ret == -1) {
        perror("camera->eqbuf");
        return -1;
    }

    return 0;
}

//停止视频采集
int camera_stop(int fd) {
    int ret;
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    ret = ioctl(fd, VIDIOC_STREAMOFF, &type);
    if (ret == -1) {
        perror("camera->stop");
        return -1;
    }
    std::cout << "camera->stop: stop capture" << std::endl;

    return 0;
}

//退出释放资源
int camera_exit(int fd) {
    int i;
    int ret = 0;
    struct v4l2_buffer vbuf;
    vbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    vbuf.memory = V4L2_MEMORY_MMAP;

    //出队所有帧缓冲
    for (i = 0; i < reqbufs.count; i++) {
        ret = ioctl(fd, VIDIOC_DQBUF, &vbuf);
        if (ret == -1)
            break;
    }

    //取消所有帧缓冲映射
    for (i = 0; i < reqbufs.count; i++)
        munmap(bufs[i].start, bufs[i].length);
    std::cout << "camera->exit: camera exit" << std::endl;
    return ret;
}

int main(int argc, char** argv) {
    int ret;
    char* jpeg_ptr = nullptr;
    unsigned int size;
    unsigned int index;

    int fd = open("/dev/video0", O_RDWR, 0);
    if (fd < 0) {
        std::cerr << "Open /dev/video0 failed" << std::endl;
        return -1;
    }
    get_capability(fd); //查看 摄像头设备的能力
    get_suppurt_video_format(fd); //查看摄像头支持的视频格式
    set_video_format(fd); //设置视频格式
    request_buf(fd); //申请帧缓冲区
    start_camera(fd); //启动采集
    camera_dqbuf(fd, (void**)&jpeg_ptr, &size, &index); //出队取一帧图像

    int pixfd = open("1.jpg", O_WRONLY | O_CREAT, 0666); //打开文件（无则 创建一个空白文件）
    write(pixfd, jpeg_ptr, size); //将一帧图像写入文件

    camera_eqbuf(fd, index); //入队归还帧缓冲
    camera_stop(fd); //关掉摄像头

    camera_exit(fd);  //退出释放资源
    close(fd);
    return 0;
}