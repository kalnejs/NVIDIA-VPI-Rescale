cmake_minimum_required(VERSION 3.10)

project(rescale)

find_package(vpi REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} vpi opencv_core opencv_highgui)
target_link_libraries(${PROJECT_NAME} opencv_imgcodecs)
