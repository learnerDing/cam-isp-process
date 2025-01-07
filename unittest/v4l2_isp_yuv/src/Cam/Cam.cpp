//Cam.cpp
#include"Cam.h"
#include "v4l2cam.h"

Cam* Cam::createCamera(const std::string& devicePath,Camtype type)
{
    switch (type)
    {
        case Cam::Camtype::V4l2:
            //return std::make_unique<V4L2Camera>(devicePath);
            std::cout<<"ok4"<<std::endl;
            return new V4L2Camera(devicePath);
            break;
        default:
            break;
    }
    return nullptr;
}
void* Cam::CameraGetRaw(const std::string& devicePath, int width, int height,Camtype type,unsigned int* sizeptr)
{
    switch (type)
    {
        case Cam::Camtype::V4l2:
            //return std::make_unique<V4L2Camera>(devicePath);
            return GetRaw(devicePath,width,height,sizeptr);//调用v4l2方式获取
            break;
        default:
            break;
    }
    return nullptr;
}