# Drone AI Vision

Learning project: computer vision for an autonomous drone, built step by step.

## What is inside

- `yolo/detect_person.py` - runs YOLO11 on an image and prints each detection with its confidence and bounding-box center.
- `yolo/blue_person2.py` - finds people in a video with YOLO11, tracks them with persistent IDs, and picks the person wearing a blue shirt by measuring blue pixels (HSV) on the torso. Outputs the horizontal offset (error) from the image center.
- `sim/track_box2.py` - PX4 + Gazebo simulation: detects a red box in the drone camera image (OpenCV) and yaws the drone toward it using MAVSDK offboard control. Typing `no` holds position, `go` resumes, `land` lands.
- `sim/detect_red.py`, `sim/run_px4.sh` - helpers.

## Tools

Python, OpenCV, Ultralytics YOLO11, PX4 Autopilot, Gazebo Sim, ROS 2 Jazzy, MAVSDK, WSL2 (Ubuntu 24.04).

## How to run (YOLO part)

    python3 -m venv venv-ai
    source venv-ai/bin/activate
    pip install ultralytics
    python yolo/detect_person.py      # needs an image named bus.jpg in the same folder
    python yolo/blue_person2.py       # needs a video named video.mp4

## Next steps

Connect the detector to the simulated drone camera, follow a chosen target ID, add obstacle avoidance.

## Credits

Ultralytics YOLO, PX4, Gazebo, ROS 2, MAVSDK, OpenCV.