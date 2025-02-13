isp_process_ncnn_encode
使用rawprovider提供虚假的raw流，经过rawQueue送入Isp，Isp将图像转化为bgr格式向后对接ncnn任务和encode任务。
结构：
isp_process_ncnn_encode
├── CMakeLists.txt
├── Encode
│   ├── EncodeThread.cpp
│   ├── yuv_encode.cpp
│   └── CMakeLists.txt
├── include
│   ├── Encode
│   │   └── EncodeThread.h
│   ├── Inference
│   │   ├── InferenceThread.h
│   │   ├── PreviewThread.h
│   │   └── yolov5.h
│   ├── Isp
│   │   ├── debayer_cuda_pipeline.h
│   │   ├── debayer.cuh
│   │   ├── isp_cuda_pipeline.h
│   │   └── isp_opencv_pipeline.h
│   └── util
│       ├── FrameQueue.h
│       ├── MatbgrToAVFrameyuv.h
│       ├── Tensor.h
│       └── Thread.h
├── Inference
│   ├── InferenceThread.cpp
│   ├── yolov5.cpp
│   ├── yolov5s_6.2.bin
│   ├── yolov5s_6.2.param
│   └── CMakeLists.txt
├── Isp
│   ├── awb
│   ├── ccm
│   ├── debayer
│   │   ├── debayer.cu
│   │   ├── debayer_cuda_pipeline.cpp
│   │   └── SimpleDebayer.cu
│   ├── gamma
│   ├── isp_cuda_pipeline.cpp
│   ├── isp_opencv_pipeline.cpp
│   └── CMakeLists.txt
├── main.cpp
├── rawprovide
│   ├── 1920_1080_8_1.raw
│   ├── 1920_1080_8_2.raw
│   ├── 1920_1080_8_3.raw
│   ├── 1920_1080_8_4.raw
│   ├── 1920_1080_8_5.raw
│   ├── CMakeLists.txt
│   ├── Raw8Provider.cpp
│   └── Raw8Provider.h
├── README.md
└── util
    ├── FrameQueue.cpp
    ├── Raw10toRaw8.cpp
    ├── Tensor.cpp
    └── Thread.cpp