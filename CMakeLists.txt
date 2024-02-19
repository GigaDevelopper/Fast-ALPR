cmake_minimum_required(VERSION 3.5)

project(Fast-ALPR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(Fast-ALPR main.cpp)

# OpenCV
find_package(OpenCV REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})

#fmt
find_package(fmt REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)

file (GLOB_RECURSE headers "workers/*.h*")
file (GLOB_RECURSE sources "workers/*.c*")

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/models ${PROJECT_BINARY_DIR})
target_sources(${PROJECT_NAME} PRIVATE ${sources} ${headers})
