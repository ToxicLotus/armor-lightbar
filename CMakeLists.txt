cmake_minimum_required(VERSION 3.10)
project(ArmorLightBar)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. 搜索外部库：OpenCV 自带 OpenCVConfig.cmake
set(OpenCV_DIR "/usr/local/opencv4.5.4/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

# 2. 指定头文件搜索路径
include_directories(${OpenCV_INCLUDE_DIRS})

# 3. 添加库链接：作业主程序
add_executable(lightbar src/main.cpp)
target_link_libraries(lightbar ${OpenCV_LIBS})

# ---------------------------------------------------------------------------
# 可选：调试/分析工具（不是作业内容，需要时才编译）
#   按需打开某个工具的注释即可编译，例如: cmake -DBUILD_TOOLS=ON ..
# ---------------------------------------------------------------------------
option(BUILD_TOOLS "build debug tools in tools/" OFF)
if(BUILD_TOOLS)
    file(GLOB TOOL_SOURCES ${CMAKE_SOURCE_DIR}/tools/*.cpp)
    foreach(src ${TOOL_SOURCES})
        get_filename_component(name ${src} NAME_WE)
        add_executable(${name} ${src})
        target_link_libraries(${name} ${OpenCV_LIBS})
    endforeach()
endif()
