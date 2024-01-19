
#!/usr/bin/env python3

# run with python filename.py -i rosbag_dir/
# "../rosbagsCar/rosbag2_2023_10_09-11_50_46"
## Links: https://stackoverflow.com/questions/73420147/how-to-read-custom-message-type-using-ros2bag

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import os


from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import random_split, Dataset


DATA_PATH = "/home/juan/ros2_ws/src/f1/dl_car_control/rosbagsCar"
LOWER_LIMIT = 0
UPPER_LIMIT = 3

class rosbagDataset(Dataset):
    def __init__(self, main_dir, transform) -> None:
        self.main_dir = main_dir
        self.transform = transform
        self.imgData = self.get_img(main_dir, "/car_img")
        self.velData = self.get_vel(main_dir, "/cmd_vel")

        self.curveLimit = 0.5
        self.dataset = self.get_dataset()
       

    def get_dataset(self):
        return self.balanceData(self.curveLimit)

    def get_img(self, rosbag_dir, topic):
        imgs = []
        with ROS2Reader(rosbag_dir) as ros2_reader:
            
            channels = 3 # Encoding = bgr8
            ros2_conns = [x for x in ros2_reader.connections]
            ros2_messages = ros2_reader.messages(connections=ros2_conns)      

            for m, msg in enumerate(ros2_messages):
                (connection, timestamp, rawdata) = msg
                    
                if (connection.topic == topic):
                    data = deserialize_cdr(rawdata, connection.msgtype)

                    # Saves the image in a readable format
                    img = np.array(data.data, dtype=data.data.dtype)
                    resizeImg = img.reshape((data.height, data.width, channels))
                    imgs.append(resizeImg)

        
        return imgs

    def get_vel(self, rosbag_dir, topic):
        vel = []

        with ROS2Reader(rosbag_dir) as ros2_reader:

            ros2_conns = [x for x in ros2_reader.connections]
            ros2_messages = ros2_reader.messages(connections=ros2_conns)
            
            for m, msg in enumerate(ros2_messages):
                (connection, timestamp, rawdata) = msg
                    
                if (connection.topic == topic):
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    linear = data.linear.x
                    angular = data.angular.z
                    vel.append([linear, angular])
        # print(vel)
        return vel

    def __len__(self):
        return len(self.velData)

    def __getitem__(self, item):
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        image_tensor = self.transform(self.dataset[item][0]).to(device)
        vel_tensor = torch.tensor(self.dataset[item][1]).to(device)

        return (vel_tensor, image_tensor)

    def balanceData(self, angular_lim):
        curve_multiplier = 1

        angular_velocities = [vel[1] for vel in self.velData]

        straight_smaple = [index for index, vel in enumerate(angular_velocities) if abs(vel) <= angular_lim]
        curve_sample = [index for index, vel in enumerate(angular_velocities) if abs(vel) > angular_lim]
        
        # Balances the numumber of curves in the dataset
        curve_aument = int(1 / (len(curve_sample) / len(self.velData)))
        n_curve = curve_multiplier * curve_aument   # Inreases the curve samples

        balanced_index = np.concatenate([curve_sample] * n_curve + [straight_smaple])
        print(len(self.imgData), "; ", len(self.velData))
        balanced_dataset = [(self.imgData[i], self.velData[i]) for i in balanced_index if i < len(self.imgData)]


        return balanced_dataset




def plotContinuousGraphic(label, vels, color, subplot):
    plt.subplot(2, 1, subplot)
    plt.plot(vels, label=label, linestyle=' ', marker='o', markersize=3, color=color)
    plt.xlabel('Sample')
    plt.ylabel('vel ' + label)
    plt.title("vel " + label)



dataset_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([66, 200]),
])


#################################################################
# Data analysis for training                                    #
#################################################################


def main():
    D = 2  # Decimales para mostrar en las trazas
    dataset_path = DATA_PATH
    all_img_data = []
    all_vel_data = []

    # Enumerar todas las carpetas dentro de dataset_path
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            print(f"Procesando rosbag en: {folder_path}")
            # Crear una instancia de tu dataset para la carpeta actual
            current_dataset = rosbagDataset(folder_path, dataset_transforms)

            # Concatenar los datos de imagen y velocidad
            all_img_data.extend(current_dataset.imgData)
            all_vel_data.extend(current_dataset.velData)

    # Crear el dataset final combinado
    final_dataset = [(all_img_data[i], all_vel_data[i]) for i in range(len(all_img_data))]

    linear_velocities = [vel[0] for vel in all_vel_data]
    angular_velocities = [vel[1] for vel in all_vel_data]

    # Plots the results
    plt.figure(figsize=(10, 6))

    plotContinuousGraphic("lineal", linear_velocities, 'b', 1)
    plotContinuousGraphic("angular", angular_velocities, 'g', 2)
    
    straight_smaples = [index for index, velocidad in enumerate(angular_velocities) if abs(velocidad) <= current_dataset.curveLimit]
    percentage = len(straight_smaples)/len(linear_velocities) * 1

    print(f"* Linear  samples => {round(percentage * 100, D)}%, mean = {round(np.mean(linear_velocities), D)}")
    print(f"* Curve samples => {round((1 - percentage) * 100, D)}%, mean = {round(np.mean(angular_velocities), D)}")

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()



    # Metodos para desajuste:

    # 1. Oversampling
    # 2. Class weighting