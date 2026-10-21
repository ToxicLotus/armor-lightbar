cmake_minimum_required(VERSION 3.10)
project(ArmorLightBar)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 搜索外部库：告诉 CMake OpenCV 的 OpenCVConfig.cmake 在哪
set(OpenCV_DIR "/usr/local/opencv4.5.4/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# 2. 指定头文件搜索路径
include_directories(${OpenCV_INCLUDE_DIRS})

# 3. 添加库链接：编译成可执行文件并链接 OpenCV
add_executable(lightbar src/main.cpp)
target_link_libraries(lightbar ${OpenCV_LIBS})

add_executable(inspect_video src/inspect_video.cpp)
target_link_libraries(inspect_video ${OpenCV_LIBS})