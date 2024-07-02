#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist

from cv_bridge import CvBridge
import ament_index_python
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
from torchvision import transforms


package_path = "/home/juan/ros2_ws/src/f1/dl_car_control"
sys.path.append(package_path + "/include")
from include.models import pilotNet
from include.Reference.pilotnet import PilotNet
from include.rosbag_preview import dataset_transforms
import cv2
import numpy as np


# Limit velocitys
MAX_ANGULAR = 2.5 
MAX_LINEAR = 20 
MIN_LINEAR = 3

DEBUG = False

C_MODEL_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/models/Specialists/Curves2.2.pth"
L_MODEL_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/models/Specialists/Lines2.2.pth"

def denormalize(normalized_value, min, max):
    return (normalized_value * (max - min)) + min


def filter_red(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = mask1 | mask2
    result = cv2.bitwise_and(image, image, mask=mask)
    
    if DEBUG:
        cv2.imshow('Filtered Red', result)
        cv2.waitKey(0)
        
    return result

def is_straight_line(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) <= 4:
        _, _, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio > 0.8 or aspect_ratio < 0.2:  # more lenient aspect ratio
            return True
    return False

def classify_line_shape(image):
    filtered_image = filter_red(image)
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    
    if DEBUG:
        cv2.imshow('Gray Image', gray)
        cv2.waitKey(0)
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    if DEBUG:
        cv2.imshow('Edges', edges)
        cv2.waitKey(0)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if DEBUG:
        debug_img = image.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 3)
        cv2.imshow('Contours', debug_img)
        cv2.waitKey(0)

    # Sort contours by length and keep only the two longest
    contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)[:2]

    line_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Ignore small contours
            if is_straight_line(contour):
                line_count += 1
    
    # Adjusting the classification logic to be more lenient
    if line_count >= 1:
        return 'LINE'
    else:
        return 'CURVE'



class carController(Node):

    def __init__(self):
        super().__init__('f1_line_follow')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.imgpublisher_ = self.create_publisher(Image, '/car_img', 10)

        self.imageSubscription = self.create_subscription(Image, '/cam_f1_left/image_raw', self.listener_callback, 10)
        self.img = None
        
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            device_type = "cuda"
        else:
            self.device = torch.device("cpu")
            device_type = "cpu"

        # Curves network
        self.c_model = PilotNet([200, 66, 3], 2).to(self.device)
        self.c_model.load_state_dict(torch.load(C_MODEL_PATH))
        
        self.c_model.to(self.device)
        
        # Lines network
        self.l_model = PilotNet([200, 66, 3], 2).to(self.device)
        self.l_model.load_state_dict(torch.load(L_MODEL_PATH))
        
        self.l_model.to(self.device)

        self.norm = False
        self.dt = 0
        self.classify = "Undetermined"


    def listener_callback(self, msg):
        bridge = CvBridge()
        self.img = bridge.imgmsg_to_cv2(msg, "bgr8")
        # self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        
        # Cut the superior (upper) half of the image
        half_height = self.img.shape[0] // 2
        bottom_half = self.img[half_height:, :, :]
        
        img = cv2.resize(bottom_half, (int(200), int(66)))
        
        shape = classify_line_shape(bottom_half)
        
        # # Mostrar la imagen
        # cv2.imshow('Imagen', bottom_half)
        # # Esperar a que el usuario presione una tecla para cerrar la ventana
        # cv2.waitKey(0)
        preprocess = transforms.Compose([
            transforms.ToTensor()
        ])
        
        img_tensor = preprocess(img).to(self.device)
        img_tensor = img_tensor.unsqueeze(0)

        # Image inference
        if shape == 'LINE':
            predictions = self.l_model(img_tensor)
            self.classify = "Line"
        else:
            predictions = self.c_model(img_tensor)
            self.classify = "Curve"
        
        v = predictions.data.cpu().numpy()[0][0]
        w = predictions.data.cpu().numpy()[0][1]
        if self.norm:
            v = denormalize(v, MIN_LINEAR, MAX_LINEAR)
            w = denormalize(w, -MAX_ANGULAR, MAX_ANGULAR)

        
        # Apply constraints to linear velocity
        linear_velocity = max(min(v, MAX_LINEAR), MIN_LINEAR)

        # Apply constraints to angular velocity
        angular_velocity = max(min(w, MAX_ANGULAR), -MAX_ANGULAR)

        # if linear_velocity == 3:
        #     # Mostrar la imagen
        #     cv2.imshow('Imagen', img)
        #     # Esperar a que el usuario presione una tecla para cerrar la ventana
        #     cv2.waitKey(0)
        
        # Velocity set
        vel_msg = Twist()
        vel_msg.linear.x = float(linear_velocity)
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = float(angular_velocity)

        # usefull for training others
        # if abs(float(vel[1])) < 0.2:
        #     vel_msg.linear.x = 9.0

        if self.dt == 10: 
            print(self.classify, " {linear speed: ", vel_msg.linear.x, "; angular speed: ", vel_msg.angular.z, "}")
            self.dt = 0
        self.dt += 1

        self.publisher_.publish(vel_msg)
        # To republish the img if you are recording a rosbag
        # self.imgpublisher_.publish(msg)

    
def main(args=None):
    rclpy.init(args=args)

    minimal_publisher = carController()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()