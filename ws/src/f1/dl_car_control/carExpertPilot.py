#!/usr/bin/env python3

## Links used
# https://github.com/ros2/rosbag2
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

import rclpy
from rclpy.node import Node
from rclpy.clock import ROSClock

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped

import numpy as np
import cv2
from cv_bridge import CvBridge

MIN_PIXEL = -360
MAX_PIXEL = 360

# Limit velocitys
MAX_ANGULAR = 4.5 # 5 Nürburgring line
MAX_LINEAR = 9 # 12 in some maps be fast
MIN_LINEAR = 1

# PID controlers
ANG_KP = 1 / 1.5
ANG_KD = 1.6 / 1.5
ANG_KI = 0.005 

LIN_KP = 1 / 1.5
LIN_KD = 1.4 / 1.5
LIN_KI = 0.001 

# Red filter parameters
color_mask = ([17, 15, 70], [50, 56, 255])

# Trace parameters
TRACE_COLOR = [0, 255, 0]
RADIUS = 2

# Distance to search the red line end
LIMIT_UMBRAL = 15

class  PID:
    def __init__(self, min, max):
        self.min = min
        self.max = max

        self.prev_error = 0
        self.int_error = 0
        # Angular values as default
        self.KP = ANG_KP
        self.KD = ANG_KD
        self.KI = ANG_KI
    
    def set_pid(self, kp, kd, ki):
        self.KP = kp
        self.KD = kd
        self.KI = ki
    
    def get_pid(self, vel):        
        
        if (vel <= self.min):
            vel = self.min
        if (vel >= self.max):
            vel = self.max
        
        self.int_error = self.int_error + vel
        dev_error = vel - self.prev_error
        self.int_error += (self.int_error + vel) 
        
        # Controls de integral value
        if (self.int_error > self.max):
            self.int_error = self.max
        if self.int_error < self.min:
            self.int_error = self.min

        self.prev_error = vel

        out = self.KP * vel + self.KI * self.int_error + self.KD * dev_error
        
        if (out > self.max):
            out = self.max
        if out < self.min:
            out = self.min
            
        return out 

class carController(Node):

    def __init__(self):
        super().__init__('f1_line_follow')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.imgpublisher_ = self.create_publisher(Image, '/car_img', 10)
        self.filteredPublisher_ = self.create_publisher(Image, '/filtered_img', 100)

        self.imageSubscription = self.create_subscription(Image, '/cam_f1_left/image_raw', self.listener_callback, 10)
        
        self.angular_pid = PID(-MAX_ANGULAR, MAX_ANGULAR)
        self.angular_pid.set_pid(ANG_KP, ANG_KD, ANG_KI)

        self.linear_pid = PID(MIN_LINEAR, MAX_LINEAR)
        self.linear_pid.set_pid(LIN_KP, LIN_KD, LIN_KI)

        self.px_rang = MAX_PIXEL - MIN_PIXEL
        self.ang_rang = MAX_ANGULAR - (- MAX_ANGULAR)
        self.lin_rang = MAX_LINEAR - MIN_LINEAR

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.filteredImage = None
        self.receivedImage = None

        self.dt = 0

    def farest_point(self, image):
        img_width = image.shape[1]
        height_mid = int(image.shape[0] / 2)
            
        x = 0
        y = 0
        count = 0
            
        for row in range (height_mid, height_mid + LIMIT_UMBRAL):
            for col in range (img_width):
                    
                comparison = image[row][col] == np.array([0, 0, 0])
                if not comparison.all():
                    y += row
                    x += col 
                    count += 1
            
        if (count == 0):
            return (0, 0)

        return [int(x / count), int(y / count)]

    def draw_traces(self, image):
        img_width = image.shape[1]
        img_height = image.shape[0]
            
        for row_index in range(img_height):
            image[row_index][int(img_width / 2)] = np.array(TRACE_COLOR)
        
    def listener_callback(self, msg):

        self.receivedImage = msg
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            
        # Apply a red filter to the image
        red_lower = np.array(color_mask[0], dtype = "uint8")
        red_upper = np.array(color_mask[1], dtype = "uint8")
        red_mask = cv2.inRange(cv_image, red_lower, red_upper)
        self.filteredImage = cv2.bitwise_and(cv_image, cv_image, mask=red_mask)
    
    
    def timer_callback(self):
        if np.any(self.filteredImage):
            width_center = self.filteredImage.shape[1] / 2
            red_farest = self.farest_point(self.filteredImage)

            # Comentar pruebas en linea recta y robustez añadida al programa
            if red_farest[0] != 0 or red_farest[1] != 0:
                
                distance = width_center - red_farest[0]
                # Pixel distance to angular vel transformation
                angular = (((distance - MIN_PIXEL) * self.ang_rang) / self.px_rang) + (-MAX_ANGULAR)
                angular_vel = self.angular_pid.get_pid(angular)

                # Inverse of angular vel and we convert it to linear velocity
                angular_inv = MAX_ANGULAR - abs(angular_vel)    # 0 - MAX_ANGULAR
                linear = (((angular_inv) * self.lin_rang) / MAX_ANGULAR) + MIN_LINEAR
                linear_vel = self.linear_pid.get_pid(linear)
                

                vel_msg = Twist()
                vel_msg.linear.x = float(linear_vel)
                vel_msg.linear.y = 0.0
                vel_msg.linear.z = 0.0
                vel_msg.angular.x = 0.0
                vel_msg.angular.y = 0.0
                vel_msg.angular.z = float(angular_vel)

                self.publisher_.publish(vel_msg)

                self.imgpublisher_.publish(self.receivedImage)

                if self.dt == 10: 
                    print("linear speed: ", linear_vel, "; angular speed: ", angular_vel)
                    self.dt = 0
                self.dt += 1

                img = CvBridge().cv2_to_imgmsg(self.filteredImage, "bgr8")
                img.header.frame_id = "your_frame_id"  # Set the frame ID here
                self.filteredPublisher_.publish(img)

            # Traces and draw image
            self.draw_traces(self.filteredImage)
            cv2.circle(self.filteredImage, red_farest, RADIUS, TRACE_COLOR, RADIUS)

            cv2.imshow("Filtered image", self.filteredImage)
            cv2.waitKey(1)




def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = carController()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()