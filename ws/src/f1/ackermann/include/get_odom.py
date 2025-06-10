#!/usr/bin/env python3

## Links used
# https://github.com/ros2/rosbag2
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import cv2
import numpy as np

# # Monmelo
# SCALE_X = -0.7195
# SCALE_Y =  0.7395
# WORLD_X0 = 605.07
# WORLD_Y0 = -211.48

# Simple
SCALE_X =  0.7448
SCALE_Y = -0.7339
WORLD_X0 = -298.65
WORLD_Y0 =  285.18


# Function to convert world coordinates to pixel
def world_to_pixel(wx, wy):
    px = int((wx - WORLD_X0) / SCALE_X)
    py = int((wy - WORLD_Y0) / SCALE_Y)
    return px, py

def pixel_to_world(px, py):
    wx = px * SCALE_X + WORLD_X0
    wy = py * SCALE_Y + WORLD_Y0
    return wx, wy


NURBU_MAP_IMG = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/Simple_circuit.png"

class carController(Node):

    def __init__(self):
        super().__init__('f1_line_follow')

        self.imageSubscription = self.create_subscription(
            Odometry,
            '/f1ros2/odom',
            self.listener_callback,
            10
        )

        self.receivedOdom = None
        self.img = cv2.imread(NURBU_MAP_IMG)

        if self.img is None:
            self.get_logger().error(f"Failed to load image at: {NURBU_MAP_IMG}")
        else:
            self.get_logger().info("Image loaded successfully.")

        # Set up mouse callback
        cv2.namedWindow("Map image")
        cv2.setMouseCallback("Map image", self.mouse_callback)

        # self.scale = 0.05  # meters/pixel

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Clicked pixel: x={x}, y={y}")
            
            wx = self.receivedOdom.pose.pose.position.x
            wy = self.receivedOdom.pose.pose.position.y
            print(f"Robot position: X = {wx:.2f} m, Y = {wy:.2f} m")
            world_x, world_y = pixel_to_world(x, y)
            print(f"Converted to world coords: X = {world_x:.2f} m, Y = {world_y:.2f} m")




    # Nurbu
    # Clicked pixel: x=349, y=125
    # Robot position: X = -372.41 m, Y = 189.34 m
    
    # Clicked pixel: x=1474, y=645
    # Robot position: X = 467.52 m, Y = -195.52 m
    
    # Montmelo
    # Clicked pixel: x=266, y=198
    # Robot position: X = 415.47 m, Y = -62.70 m
    # Clicked pixel: x=1416, y=326
    # Robot position: X = -419.49 m, Y = 23.45 m
    # Converted to world coords: X = 423.91 m, Y = 40.07 m
    # Clicked pixel: x=656, y=71
    # Robot position: X = 134.25 m, Y = -157.61 m
    
    # SIMPLE
    # Clicked pixel: x=760, y=460
    # Robot position: X = 267.40 m, Y = -52.41 m
    # Clicked pixel: x=577, y=174
    # Robot position: X = 131.10 m, Y = 157.48 m

    # def pixel_to_world(self, px, py):
    #     # Clicked pixel: x=313, y=128
    #     # Robot position: X = -372.45 m, Y = 189.36 m

    #     # Clicked pixel: x=1491, y=656
    #     # Robot position: X = 467.07 m, Y = -195.34 m
        
    #     # Clicked pixel: x=811, y=367
    #     # Robot position: X = -0.12 m, Y = -0.05 m
        
    #     # m/pxl
    #     scale_x = 0.712
    #     scale_y = -0.7286  # Negative because image Y increases downward
    #     world_px0 = 811
    #     world_py0 = 367
        
    #     world_x = (px - world_px0) * scale_x
    #     world_y = (py - world_py0) * scale_y
        
    #     print(world_x, px - world_px0, px)

    #     return world_x, world_y

    def listener_callback(self, msg):
        self.receivedOdom = msg

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        # print(f"Robot position: X = {x:.2f} m, Y = {y:.2f} m")

        if self.img is not None:
            px, py = world_to_pixel(x, y)
            img_copy = self.img.copy()
            cv2.circle(img_copy, (px, py), 5, (0, 255, 0), -1)  # Green dot
            cv2.imshow("Map image", img_copy)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = carController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
