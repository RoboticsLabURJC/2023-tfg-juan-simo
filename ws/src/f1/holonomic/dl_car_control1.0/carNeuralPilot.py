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
MAX_ANGULAR = 4.5 
MAX_LINEAR = 20 
MIN_LINEAR = 3

MODEL_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/models/E2_2/pilot_net_model_121_afine.ckpt"

def load_checkpoint(model: pilotNet, optimizer: optim.Optimizer = None):
    checkpoint = torch.load(MODEL_PATH)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        

def denormalize(normalized_value, min, max):
    return (normalized_value * (max - min)) + min


class carController(Node):

    def __init__(self):
        super().__init__('f1_line_follow')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.imgpublisher_ = self.create_publisher(Image, '/car_img', 10)

        self.imageSubscription = self.create_subscription(Image, '/cam_f1_left/image_raw', self.listener_callback, 10)
        self.img = None

        # Neural network
        self.model = PilotNet([200, 66, 3], 2)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        
        
        # self.model = pilotNet()
        # load_checkpoint(self.model)

        self.device = torch.device("cuda:0")
        self.model.to(self.device)

        self.norm = True
        self.dt = 0


    def listener_callback(self, msg):
        bridge = CvBridge()
        self.img = bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Cut the superior (upper) half of the image
        half_height = self.img.shape[0] // 2
        bottom_half = self.img[half_height:, :, :]
        
        img = cv2.resize(bottom_half, (int(200), int(66)))
        
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
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        vel = predictions[0].tolist()
        if self.norm:
            v = denormalize(vel[0], MIN_LINEAR, MAX_LINEAR)
            w = denormalize(vel[1], -MAX_ANGULAR, MAX_ANGULAR)
        else:
            v = vel[0]
            w = vel[1]
        
        # Apply constraints to linear velocity
        linear_velocity = max(min(v, MAX_LINEAR), MIN_LINEAR)

        # Apply constraints to angular velocity
        angular_velocity = max(min(w, MAX_ANGULAR), -MAX_ANGULAR)

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
            print("linear speed: ", vel_msg.linear.x, "; angular speed: ", vel_msg.angular.z)
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