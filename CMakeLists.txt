cmake_minimum_required(VERSION 3.10)
project(demo)

set(CMAKE_CXX_STANDARD 14)
SET(CMAKE_BUILD_TYPE Release)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")

set(ncnn_DIR "/home/caia/lc_ws/ncnn/build/install/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")

find_package(OpenCV REQUIRED)
find_package(ncnn REQUIRED)

add_executable(demo main.cpp tracking.cpp)
target_link_libraries(demo
        ncnn
        ${OpenCV_LIBS}
        -fopenmp
        )


add_executable(demo_rgb main_bgr.cpp tracking.cpp)
target_link_libraries(demo_rgb
        ncnn
        ${OpenCV_LIBS}
        -fopenmp
        )

add_executable(demo_gray main_gray.cpp tracking.cpp)
target_link_libraries(demo_gray
        ncnn
        ${OpenCV_LIBS}
        -fopenmp
        )