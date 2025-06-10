#!/usr/bin/env python3

import os
import cv2
import csv
import numpy as np
import torch
from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor

DATA_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/rosbagsCar"
CSV_FILE_PATH = "output.csv"  # Define your CSV file name or path here

class RosbagDataset(Dataset):
    def __init__(self, main_dir, transform) -> None:
        self.main_dir = main_dir
        self.transform = transform
        self.imgData, self.imgPaths = self.get_img(main_dir, "/car_img")
        self.velData = self.get_vel(main_dir, "/cmd_vel")
        self.len = len(self.imgPaths)
        self.curveLimit = 0.15

    def get_img(self, rosbag_dir, topic):
        imgs = []
        imgs_paths = []
        image_dir = os.path.join(self.main_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        with ROS2Reader(rosbag_dir) as ros2_reader:
            channels = 3  # Encoding = bgr8
            for connection, timestamp, rawdata in ros2_reader.messages():
                if connection.topic == topic:
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    img = np.frombuffer(data.data, dtype=np.uint8)
                    resizeImg = img.reshape((data.height, data.width, channels))
                    imgs.append(resizeImg)
                    
                    img_path = os.path.join(image_dir, f"{timestamp}.png")
                    cv2.imwrite(img_path, resizeImg)
                    imgs_paths.append(img_path)

        return imgs, imgs_paths

    def get_vel(self, rosbag_dir, topic):
        vel = []
        with ROS2Reader(rosbag_dir) as ros2_reader:
            for connection, timestamp, rawdata in ros2_reader.messages():
                if connection.topic == topic:
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    linear = data.linear.x
                    angular = data.angular.z
                    vel.append([linear, angular])
        return vel

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        image_tensor = self.transform(self.imgData[item])
        vel_tensor = torch.tensor(self.velData[item])
        return image_tensor, vel_tensor

def generate_csv(img_paths, velocities, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['Image Path', 'Linear Velocity', 'Angular Velocity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for img_path, velocity in zip(img_paths, velocities):
            writer.writerow({'Image Path': img_path, 'Linear Velocity': velocity[0], 'Angular Velocity': velocity[1]})

def main():
    dataset_transforms = Compose([
        ToTensor(),
        Resize([66, 200], antialias=True),
    ])

    dataset = RosbagDataset(DATA_PATH, dataset_transforms)
    generate_csv(dataset.imgPaths, dataset.velData, CSV_FILE_PATH)
    print(f"CSV file has been created at {CSV_FILE_PATH}")

if __name__ == "__main__":
    main()
