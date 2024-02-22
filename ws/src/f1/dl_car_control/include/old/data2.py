
#!/usr/bin/env python3

# run with python filename.py -i rosbag_dir/
# "../rosbagsCar/rosbag2_2023_10_09-11_50_46"
## Links: https://stackoverflow.com/questions/73420147/how-to-read-custom-message-type-using-ros2bag

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

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
        self.imgData, self.imgTimestamps = self.get_img(main_dir, "/cam_f1_left/image_raw")
        self.velData, self.velTimestamps = self.get_vel(main_dir, "/cmd_vel")

        self.max_time_diff = 0.1
        self.curveLimit = 0.5
        self.dataset = self.get_dataset(self.max_time_diff)

    def get_img(self, rosbag_dir, topic):
        imgs = []
        timestamps = []
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
                    timestamps.append(timestamp)

        return imgs, timestamps

    def get_vel(self, rosbag_dir, topic):
        vel = []
        timestamps = []

        with ROS2Reader(rosbag_dir) as ros2_reader:

            ros2_conns = [x for x in ros2_reader.connections]
            ros2_messages = ros2_reader.messages(connections=ros2_conns)
            
            for m, msg in enumerate(ros2_messages):
                (connection, timestamp, rawdata) = msg
                    
                if (connection.topic == topic):
                    data = deserialize_cdr(rawdata, connection.msgtype)
                    vel.append([data.linear.x, data.angular.z])
                    timestamps.append(timestamp)

        return vel, timestamps
    
    def get_closest_vel_index(self, img_timestamp, max_time_diff):
        # Encuentra el índice del comando de velocidad más cercano en tiempo
        # Retorna None si la diferencia de tiempo excede el umbral máximo
        closest_index = None
        closest_time_diff = float('inf')

        for i, vel_timestamp in enumerate(self.velTimestamps):
            time_diff = abs(vel_timestamp - img_timestamp)
            if time_diff < closest_time_diff:
                closest_index = i
                closest_time_diff = time_diff
                print(f"Comparando img_timestamp {img_timestamp} con vel_timestamp {vel_timestamp}, diferencia: {time_diff}")

        if closest_time_diff > max_time_diff:
            return None
        else:
            return closest_index

    def get_dataset(self, max_time_diff):  # max_time_diff es el umbral de tiempo en segundos
        matched_data = []
        for img, img_timestamp in zip(self.imgData, self.imgTimestamps):
            vel_index = self.get_closest_vel_index(img_timestamp, max_time_diff)
            if vel_index is not None:
                matched_data.append((img, self.velData[vel_index]))

        return self.balanceData(self.curveLimit, matched_data)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, velocity = self.dataset[idx]
        image_tensor = self.transform(image)
        vel_tensor = torch.tensor(velocity)
        return image_tensor, vel_tensor


    def balanceData(self, angular_lim, matched_data):
        curve_multiplier = 1

        # Extraer las velocidades angulares de matched_data
        angular_velocities = [data[1][1] for data in matched_data]  # [1][1] para obtener la velocidad angular

        # Determinar los índices para muestras rectas y curvas
        straight_sample = [index for index, vel in enumerate(angular_velocities) if abs(vel) <= angular_lim]
        curve_sample = [index for index, vel in enumerate(angular_velocities) if abs(vel) > angular_lim]

        # Evitar división por cero si no hay curvas
        if len(curve_sample) == 0:
            curve_aument = 0
        else:
            curve_aument = int(1 / (len(curve_sample) / len(angular_velocities)))
        n_curve = curve_multiplier * curve_aument

        # Crear índices balanceados
        balanced_index = np.concatenate([curve_sample] * n_curve + [straight_sample])
        
        # Construir el dataset balanceado
        balanced_dataset = [matched_data[i] for i in balanced_index if i < len(matched_data)]

         # Trazas para verificar el tamaño de los conjuntos de datos de imágenes y velocidades
        print("Cantidad de imágenes:", len(self.imgData))
        print("Cantidad de comandos de velocidad:", len(self.velData))
        print("Tamaño del conjunto de datos emparejado:", len(matched_data))

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
    D = 2  # Decimals to show in the traces
    dataset = rosbagDataset(DATA_PATH, dataset_transforms)

    vels = [velocitys for image, velocitys in dataset.dataset]
    linear_velocities = [vel[0] for vel in vels]
    angular_velocities = [vel[1] for vel in vels]

    # Plots the results
    plt.figure(figsize=(10, 6))

    plotContinuousGraphic("lineal", linear_velocities, 'b', 1)
    plotContinuousGraphic("angular", angular_velocities, 'g', 2)
    
    straight_smaples = [index for index, velocidad in enumerate(angular_velocities) if abs(velocidad) <= dataset.curveLimit]
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