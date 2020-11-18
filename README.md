# YolactROS
simplified ROS wrapper for Yolact, Yolact++, this wrapper doesn't publish bbox or mask rather it simply subscribes to realsense color/image_raw topic and display segmentation usin OpenCV

****
* tested on Ubuntu 18.04, ROS Melodic, RTX 2080-Ti, CUDA 10.1, Python3.7, PyTorch 1.4.1
* git clone in your catkin_ws https://github.com/avasalya/YolactROS.git
* refer `environment.yml` for other anaconda packages

## adapted from
* https://github.com/dbolya/yolact
* https://github.com/Eruvae/yolact_ros

## create conda environment
* `conda env create -f environment.yml`

## install realsense ROS package
* https://github.com/IntelRealSense/realsense-ros

<!-- ## download and unzip `txonigiri` within main directory
* https://www.dropbox.com/sh/wkmqd0w1tvo4592/AADWt9j5SjiklJ5X0dpsSILAa?dl=0 -->


<!-- <br /> -->

# RUN
### 1. launch camera
* `roslaunch realsense2_camera rs_rgbd.launch align_depth:=true`

### 2. launch yolact node
* rosrun yolact_ros yolact_ros.py
* or simply `python3 scripts/yolact_ros.py`