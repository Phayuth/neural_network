# Torch Implementation with ROS C++ and Python

# Installation 
- Cuda : https://www.youtube.com/watch?v=UhuK9ShIpf8
- Cudnn : https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#installlinux
- Torchlib : 
	- https://pytorch.org/cppdocs/installing.html
	- https://pytorch.org/get-started/locally/


# System
- Ubuntu 18.04 LTS
- GPU : NVIDIA RTX 3070
- RAM : 32 GBS
- NVIDIA Driver : 520.61.05
- CUDA Version : 11.8
- CUDA Capability : 8.6 (86)
- Libtorch Library Version : Stable 1.8.2 ABI - libtorch-cxx11-abi-shared-with-deps-1.8.2+cu111
- CUDA library : cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64
- CUDNN library : cudnn-linux-x86_64-8.6.0.163_cuda11-archive


### References 
- Torchlib tutorial : https://www.youtube.com/watch?v=RFq8HweBjHA
- Some source code : https://github.com/ActiveIntelligentSystemsLab/pytorch_enet_ros
- catkin_lib and torch lib error : 
	- https://answers.ros.org/question/347885/combining-cmakeliststxt-of-libtorch-and-cmakeliststxt-of-ros-package
	- https://github.com/pytorch/pytorch/issues/49450
	- https://stackoverflow.com/questions/61438764/combining-two-cmakelists-txt-file-ros-and-libtorch
	- https://answers.ros.org/question/239690/how-to-include-a-library-in-cmakelists/