cmake_minimum_required(VERSION 3.5)

project(Fast-ALPR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Fast-ALPR main.cpp
    yolov8/yolovpose.h yolov8/yolovpose.cpp
    yolov8/keypoints.h yolov8/keypoints.cpp
    yolov8/boundboxe.h yolov8/boundboxe.cpp
    tools/imgutils.h tools/imgutils.cpp
    ocr/clipper.h ocr/clipper.cpp
    ocr/cls_procces.h ocr/cls_procces.cpp
    ocr/crnn_process.h ocr/crnn_process.cpp
    ocr/db_post_procces.h ocr/db_post_procces.cpp
)

# OpenCV
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

#fmt
find_package(fmt REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)

file (GLOB_RECURSE headers "yolov8/*.h*" "tools/*.h*" "ocr/*.h*")
file (GLOB_RECURSE sources "yolov8/*.c*" "tools/*.c*" "ocr/*.c*")

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/models ${PROJECT_BINARY_DIR})
target_sources(${PROJECT_NAME} PRIVATE ${sources} ${headers})
