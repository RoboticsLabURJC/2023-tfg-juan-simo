#!/usr/bin/env python3


from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
import numpy as np
import cv2
import csv
import os
from data_loader import *

import matplotlib.pyplot as plt

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
MIN_LINEAR = 3

# PID controlers
ANG_KP = 0.667
ANG_KD = 1.067
ANG_KI = 0.001 

LIN_KP = 1 / 1.5
LIN_KD = 1.4 / 1.5
LIN_KI = 0.001 

# Red filter parameters
color_mask = ([17, 15, 70], [50, 56, 255])

# Trace parameters
TRACE_COLOR = [0, 255, 0]
RADIUS = 2

# Distance to search the red line end
LIMIT_UMBRAL = 22

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

CSV_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/Simple_E2"


def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")
        
def farest_point(image):
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
            
    start_row = height_mid
    end_row = height_mid + LIMIT_UMBRAL        
    # Add a green horizontal line at the first and last row evaluated
    if count > 0:
        image[start_row, :] = [0, 255, 0]  # Set the first row to green
        image[end_row - 1, :] = [0, 255, 0]  # Set the last row to green
            
    if (count == 0):
        return (0, 0)

    return [int(x / count), int(y / count)]
        
straight = 0

def vels(img_name, angular_pid, linear_pid):
    img = cv2.imread(img_name)    
    
    px_rang = MAX_PIXEL - MIN_PIXEL
    ang_rang = MAX_ANGULAR - (- MAX_ANGULAR)
    lin_rang = MAX_LINEAR - MIN_LINEAR
    global straight
    
    width_center = img.shape[1] / 2
    red_farest = farest_point(img)

    # Comentar pruebas en linea recta y robustez añadida al programa
    if red_farest[0] != 0 or red_farest[1] != 0:
    
        distance = width_center - red_farest[0]
        # Pixel distance to angular vel transformation
        angular = (((distance - MIN_PIXEL) * ang_rang) / px_rang) + (-MAX_ANGULAR)
        angular_vel = angular_pid.get_pid(angular)

        # Inverse of angular vel and we convert it to linear velocity
        # angular_inv = MAX_ANGULAR - abs(angular_vel)    # 0 - MAX_ANGULAR
        # linear = (((angular_inv) * lin_rang) / MAX_ANGULAR) + MIN_LINEAR
        # linear_vel = linear_pid.get_pid(linear)
        
        # linear_vel = linear_vel*0.8
        # if abs(angular_vel) < 0.2:
        #     straight += 1
        # else:
        #     straight = 0
        
        # if straight >= 5:   
        #     if abs(angular_vel) < 0.1:
        #         linear_vel = linear_vel*2.2
        #     elif abs(angular_vel) < 0.15:
        #         linear_vel = linear_vel*1.9
        #     elif abs(angular_vel) < 0.2:
        #         linear_vel = linear_vel*1.7
        #     elif abs(angular_vel) < 0.22:
        #         linear_vel = linear_vel*1.5
        #     elif abs(angular_vel) < 0.25:
        #         linear_vel = linear_vel
        #     else:
        #         linear_vel = linear_vel*0.8
            
        #     linear_vel = linear_vel + straight/10



        # if dt == 10: 
        #     print("linear speed: ", linear_vel, "; angular speed: ", angular_vel)
        #     dt = 0
        # dt += 1
        
        return angular_vel

        

        

def main():
    angular_pid = PID(-MAX_ANGULAR, MAX_ANGULAR)
    angular_pid.set_pid(ANG_KP, ANG_KD, ANG_KI)

    linear_pid = PID(MIN_LINEAR, MAX_LINEAR)
    linear_pid.set_pid(LIN_KP, LIN_KD, LIN_KI)
    
    labels = []
    all_images, all_data = load_data(CSV_PATH)
    labels = parse_csv(all_data, labels)
    
    for i in tqdm(range(len(labels))):
        e_w = vels(all_images[i], angular_pid, linear_pid)
        d_w = labels[i][1]
                
        comp = e_w * d_w
        
        if comp < 0:
            print("WRONG, EXPERT = ", e_w, "; DATASET = ", d_w)
            img = cv2.imread(all_images[i])
            # Mostrar la imagen
            cv2.imshow('Imagen', img)
            # Esperar a que el usuario presione una tecla para cerrar la ventana
            cv2.waitKey(0)        
            
            
        
        
        
    
     
    

if __name__ == "__main__":
    main()


    # Metodos para desajuste:

    # 1. Oversampling
    # 2. Class weighting