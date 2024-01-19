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
MODEL_PATH_A = "/home/juan/ros2_ws/src/f1/dl_car_control/models/2agressive.tar"
MODEL_PATH_S = "/home/juan/ros2_ws/src/f1/dl_car_control/models/selection.tar"


def load_checkpoint(path, model: pilotNet, optimizer: optim.Optimizer = None):
    checkpoint = torch.load(path)

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
        self.model_save = pilotNet()
        load_checkpoint(MODEL_PATH, self.model_save)
        self.model_agr = pilotNet()
        load_checkpoint(MODEL_PATH_A, self.model_agr)
        self.model_sel = pilotNet()
        load_checkpoint(MODEL_PATH_S, self.model_sel)

        self.device = torch.device("cuda:0")
        self.model_save.to(self.device)
        self.model_agr.to(self.device)
        self.model_sel.to(self.device)

        self.dt = 0


    def listener_callback(self, msg):
        bridge = CvBridge()
        self.img = bridge.imgmsg_to_cv2(msg, "bgr8")
        img_tensor = dataset_transforms(self.img).to(self.device)
        img_tensor = img_tensor.unsqueeze(0)

        # Image inference
        with torch.no_grad():
            predictions_s = self.model_save(img_tensor)
            predictions_a = self.model_agr(img_tensor)
            model_predict = self.model_sel(img_tensor)

        # Velocity set
        vel_s = predictions_s[0].tolist()
        vel_a = predictions_a[0].tolist()
        model_to_use = model_predict[0].tolist()[0]

        vel_msg = Twist()

        if model_to_use < 1.75:
            vel_msg.angular.z = float(vel_a[1])
            vel_msg.linear.x = float(vel_a[0])
            model_used = "agro"
        else:
            vel_msg.angular.z = float(vel_s[1])
            vel_msg.linear.x = float(vel_s[0])
            model_used = "save"


        vel_msg.linear.y = 0.0
        vel_msg.linear.z = 0.0
        vel_msg.angular.x = 0.0
        vel_msg.angular.y = 0.0
        

        if self.dt == 10: 
            print("linear speed: ", vel_msg.linear.x, "; angular speed: ", vel_msg.angular.z, "; model = ", model_used)
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