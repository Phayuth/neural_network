cmake_minimum_required(VERSION 3.2)

project(torchcpp)

# set(CMAKE_PREFIX_PATH /home/airlab/pytch_cpp/resource/libtorch/libtorch) #link to where we save libtorch

# using Lastest version with ABI
set(CMAKE_PREFIX_PATH /home/airlab/pytch_cpp/libtorch/libtorch) #link to where we save libtorch

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)