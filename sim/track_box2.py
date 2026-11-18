import asyncio
import threading
import time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

# Gazebo-র camera-র topic
TOPIC = "/world/default/model/x500_mono_cam_0/link/camera_link/sensor/camera/image"

KP = 0.08          # error (pixel) কে ঘোরার গতিতে (deg/s) রূপান্তরের গুণক
MAX_YAW = 30.0     # সর্বোচ্চ ঘোরার গতি
DEAD_ZONE = 20     # error এর চেয়ে কম হলে থেমে যাবে
RUN_SECONDS = 600  # সর্বোচ্চ কতক্ষণ চলবে

# দুটো আলাদা অংশ (ছবি পড়া আর কমান্ড নেওয়া) এই বাক্স দিয়ে একে অপরের সাথে কথা বলে
shared = {"x": None, "w": 1280, "t": 0.0, "mode": "follow", "quit": False}


class Detector(Node):
    """Camera-র ছবি পড়ে লাল box খোঁজে।"""

    def __init__(self):
        super().__init__("box_tracker")
        self.create_subscription(Image, TOPIC, self.cb, 1)

    def cb(self, msg):
        # ছবির সংখ্যাগুলোকে (রঙের মান) বড় একটা তালিকা থেকে ছবির আকারে সাজাই
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        # লাল রঙ HSV-তে দুই প্রান্তে থাকে, তাই দুটো সীমা
        m1 = cv2.inRange(hsv, (0, 120, 40), (10, 255, 255))
        m2 = cv2.inRange(hsv, (170, 120, 40), (180, 255, 255))
        mask = cv2.bitwise_or(m1, m2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 50]
        if contours:
            c = max(contours, key=cv2.contourArea)       # সবচেয়ে বড় লাল জায়গা
            x, y, w, h = cv2.boundingRect(c)
            shared["x"] = x + w // 2                      # box-এর মাঝখানের x
        else:
            shared["x"] = None                            # box দেখা যাচ্ছে না
        shared["w"] = msg.width
        shared["t"] = time.time()


def ros_thread():
    rclpy.init()
    rclpy.spin(Detector())


def command_thread():
    """তোমার লেখা কমান্ড (no / go / land) পড়ে।"""
    print("Commands (then press Enter):  no = stop following,  go = follow again,  land = finish")
    while not shared["quit"]:
        try:
            cmd = input().strip().lower()
        except EOFError:
            break
        if cmd in ("no", "n", "stop"):
            shared["mode"] = "hold"
            print(">>> NO order received: holding position")
        elif cmd in ("go", "g", "follow"):
            shared["mode"] = "follow"
            print(">>> GO: following again")
        elif cmd in ("land", "l", "quit", "q"):
            shared["quit"] = True
            print(">>> LAND order received")
        elif cmd:
            print(">>> unknown command. Use: no / go / land")


async def run():
    threading.Thread(target=ros_thread, daemon=True).start()

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14540")
    async for state in drone.core.connection_state():
        if state.is_connected:
            break
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            break

    print("arming")
    await drone.action.arm()
    print("takeoff")
    await drone.action.takeoff()

    # ২ মিটার উচ্চতায় ওঠা পর্যন্ত অপেক্ষা (simulation ধীর, তাই সময় নির্দিষ্ট না ধরে উচ্চতা দেখি)
    alt = {"v": 0.0}

    async def watch_alt():
        async for pos in drone.telemetry.position():
            alt["v"] = pos.relative_altitude_m

    asyncio.create_task(watch_alt())
    for _ in range(60):
        if alt["v"] > 2.0:
            break
        await asyncio.sleep(2)

    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    try:
        await drone.offboard.start()
    except OffboardError as e:
        print(f"offboard failed: {e}")
        await drone.action.land()
        return

    threading.Thread(target=command_thread, daemon=True).start()
    print("tracking... type no / go / land, then Enter")
    start = time.time()
    last_print = 0.0
    try:
        while not shared["quit"] and time.time() - start < RUN_SECONDS:
            x = shared["x"]
            fresh = (time.time() - shared["t"]) < 3.0    # ছবি কি সদ্য এসেছে?
            yaw = 0.0
            if shared["mode"] == "hold":
                status = "HOLD (type go to follow again)"
            elif x is not None and fresh:
                error = x - shared["w"] / 2               # box মাঝখান থেকে কত pixel দূরে
                if abs(error) > DEAD_ZONE:
                    yaw = max(-MAX_YAW, min(MAX_YAW, KP * error))
                status = f"FOLLOW box x={x}, error={error:.0f}, yaw={yaw:.1f} deg/s"
            else:
                status = "FOLLOW: box not visible, holding"
            if time.time() - last_print > 2.0:
                print(status)
                last_print = time.time()
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, yaw))
            await asyncio.sleep(0.5)
    finally:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
        try:
            await drone.offboard.stop()
        except OffboardError:
            pass
        print("landing")
        await drone.action.land()


asyncio.run(run())