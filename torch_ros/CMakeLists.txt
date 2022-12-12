cmake_minimum_required(VERSION 3.0.2)
project(ros_torch)


# find the path to the torch library folder that contain cmake list TorchConfig.cmake
# then in there, we get more additional param such as :
#   TORCH_FOUND        -- True if the system has the Torch library
#   TORCH_INCLUDE_DIRS -- The include directories for torch
#   TORCH_LIBRARIES    -- Libraries to link against
#   TORCH_CXX_FLAGS    -- Additional (required) compiler flags
set(Torch_DIR "/home/airlab/pytch_cpp/libtorch/libtorch/share/cmake/Torch")

# find package for ROS system
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)

# find package for external library such as Torch
find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ros_torch
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)


include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)

add_executable(main src/main.cpp)
target_link_libraries(main ${catkin_LIBRARIES} ${TORCH_LIBRARIES})
# set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 14)