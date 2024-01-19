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
from models import pilotNet
from data import dataset_transforms, DATA_PATH


# Limit velocitys
MAX_ANGULAR = 4.5 
MAX_LINEAR = 9 
MIN_LINEAR = 1

MODEL_PATH = "/home/juan/ros2_ws/src/f1/dl_car_control/models/first_model.tar"
MODEL_PATH_A = "/home/juan/ros2_ws/src/f1/dl_car_control/models/agressive.tar"

def load_checkpoint(model: pilotNet, optimizer: optim.Optimizer = None):
    checkpoint = torch.load(MODEL_PATH)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

class carController(Node):

    def __init__(self):
        super().__init__('f1_line_follow')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.imgpublisher_ = self.create_publisher(Image, '/car_img', 10)

        self.imageSubscription = self.create_subscription(Image, '/cam_f1_left/image_raw', self.listener_callback, 10)
        self.img = None

        # Neural network
        self.model = pilotNet()
        load_checkpoint(self.model)

        self.device = torch.device("cuda:0")
        self.model.to(self.device)

        self.dt = 0


    def listener_callback(self, msg):
        bridge = CvBridge()
        self.img = bridge.imgmsg_to_cv2(msg, "bgr8")
        img_tensor = dataset_transforms(self.img).to(self.device)
        img_tensor = img_tensor.unsqueeze(0)

        # Image inference
        with torch.no_grad():
            predictions = self.model(img_tensor)

        # Velocity set
        vel = predictions[0].tolist()
        vel_msg = Twist()
        vel_msg.linear.x = float(vel[0])
        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        vel_msg.angular.z = float(vel[1])

        # usefull for training others
        if abs(float(vel[1])) < 0.2:
            vel_msg.linear.x = 9.0

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