import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image

TOPIC = "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"

class Detector(Node):
    def __init__(self):
        super().__init__("red_detector")
        self.create_subscription(Image, TOPIC, self.cb, 10)

    def cb(self, msg):
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        m1 = cv2.inRange(hsv, (0, 120, 40), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 120, 40), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 50]

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)
            print(f"RED OBJECT FOUND: center=({x + w//2}, {y + h//2}), size={w}x{h}, area={int(cv2.contourArea(c))}")
        else:
            print("no red object found")

        cv2.imwrite("/home/absms_shakib/detected.png", bgr)
        print("saved: ~/detected.png")
        raise SystemExit

rclpy.init()
try:
    rclpy.spin(Detector())
except SystemExit:
    pass
rclpy.shutdown()
