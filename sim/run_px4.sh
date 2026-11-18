#!/bin/bash
# PX4 + Gazebo চালানোর আগে সঠিক gz ঠিক করে নেয়
unset GZ_CONFIG_PATH
export PATH=/usr/bin:$PATH
cd ~/PX4-Autopilot
make px4_sitl gz_x500_mono_cam
