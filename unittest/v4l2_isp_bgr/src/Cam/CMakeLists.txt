cmake_minimum_required(VERSION 3.10)
project(cameratest)

add_executable(camera Camtest.cpp v4l2cam.cpp Cam.cpp)