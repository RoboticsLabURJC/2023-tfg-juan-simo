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

CENTER = 150

C_MODEL_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/models/Specialists/Curves.pth"
L_MODEL_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/models/Specialists/Lines.pth"

def denormalize(normalized_value, min, max):
    return (normalized_value * (max - min)) + min

def clasify(img):
    img_width = img.shape[1]  # 640
    img_height = img.shape[0] # 240
        
    height_mid = int(img_height / 2)
    width_mid = int(img_width / 2)
    count = 0
    y = 0
    
    
    
    # Apply a red filter to the image
    color_mask = ([17, 15, 70], [50, 56, 255])
    red_lower = np.array(color_mask[0], dtype = "uint8")
    red_upper = np.array(color_mask[1], dtype = "uint8")
    red_mask = cv2.inRange(img, red_lower, red_upper)
    image = cv2.bitwise_and(img, img, mask=red_mask)
                
    # print(img_height, " ", img_width)
        
        
            
    for row_index in range(img_width):
        # image[int(160)][row_index] = np.array([0, 255, 0])
        # image[int(31)][row_index] = np.array([0, 255, 0])
            
        comparison = image[img_height - 1][row_index] == np.array([0, 0, 0])
        if not comparison.all():
            count += 1
            y += row_index
                
    if count == 0:
        return False
        
    mid = int(y/count)
        
    # for colum in range(img_height):  
    #     # if(colum > img_height/2):
    #     #     image[colum][int((img_width / 2) + CENTER)] = np.array([0, 255, 0])
    #     #     image[colum][int((img_width / 2) - CENTER)] = np.array([0, 255, 0])
    #     # else:
    #     #     image[colum][int((img_width / 2) + 30)] = np.array([0, 255, 0])
    #     #     image[colum][int((img_width / 2) - 30)] = np.array([0, 255, 0])
                
    #     image[colum][mid] = np.array([0, 255, 0])
    #     image[colum][mid - 59] = np.array([0, 255, 0])
    #     image[colum][mid + 59] = np.array([0, 255, 0])
    

    # # Mostrar la imagen
    # cv2.imshow('Imagen', image)
    # # Esperar a que el usuario presione una tecla para cerrar la ventana
    # cv2.waitKey(1)
          
            
    for colum in range(0, 30):
        if ((mid - CENTER) <= 0) or ((mid + CENTER) >= img_width):
            return False
                            
        for row in range(0, mid - CENTER):
            comparison = image[colum][row] == np.array([0, 0, 0])
            if not comparison.all():
                # cv2.circle(image, [row, colum], 3, [0, 255, 0], 3)
                return False
                
        for row in range(mid + CENTER, img_width):
            comparison = image[colum][row] == np.array([0, 0, 0])
            if not comparison.all():
                # cv2.circle(image, [row, colum], 3, [0, 255, 0], 3)
                return False
                
        
        
    return True


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
        
        line = clasify(img)
        
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
        if line:
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