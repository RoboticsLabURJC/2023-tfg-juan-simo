
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
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import torch
from torch.utils.data import random_split, Dataset


DATA_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/rosbagsCar/Simple_ACK"

class rosbagDataset(Dataset):
    def __init__(self, main_dir, transform) -> None:
        self.len = 0
        self.main_dir = main_dir
        self.transform = transform
        self.imgData = self.get_img(main_dir, "/car_img")
        self.velData = self.get_vel(main_dir, "/f1ros2/cmd_vel")

        self.curveLimit = 0.15
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
                    
                    # Cut the superior (upper) half of the image
                    half_height = resizeImg.shape[0] // 2
                    bottom_half = resizeImg[half_height:, :, :]
                    imgs.append(bottom_half)
                   
                   
                    # # Display the image using matplotlib
                    # plt.imshow(bottom_half)  # Convert BGR to RGB
                    # plt.show()
        
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
        return self.len

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
        # p_curve_sample = [index for index, vel in enumerate(angular_velocities) if ((abs(vel) > angular_lim) and (vel >= 0))]
        # n_curve_sample = [index for index, vel in enumerate(angular_velocities) if ((abs(vel) > angular_lim) and (vel <= 0))]
        
        # p_len = len(p_curve_sample)
        # n_len = len(n_curve_sample)
        # if p_len > n_len:
        #     curve_balance = int(n_len / p_len)
        #     curve_sample = np.concatenate([p_curve_sample] + [n_curve_sample] * curve_balance)
        # else:
        #     curve_balance = int(n_len / p_len )
        #     curve_sample = np.concatenate([p_curve_sample] * curve_balance + [n_curve_sample])
        
        
        
        # Balances the numumber of curves in the dataset
        # curve_aument = int(1 / (len(curve_sample) / len(self.velData)))
        # n_curve = curve_multiplier * curve_aument   # Inreases the curve samples
        n_curve = 1
        
        balanced_index = np.concatenate([curve_sample] * n_curve + [straight_smaple])
        print(len(self.imgData), "; ", len(self.velData))
        balanced_dataset = [(self.imgData[i], self.velData[i]) for i in balanced_index]
        dataset_len = len(balanced_dataset)
        print(dataset_len)

        # unbalanced_index = np.concatenate([straight_smaple]*2)
        # dataset_len = len(unbalanced_index)
        # print(dataset_len, "; Unbalanced data")
        # unbalanced_dataset = [(self.imgData[i], self.velData[i]) for i in unbalanced_index if i < len(self.imgData)]

        self.len = dataset_len

        return balanced_dataset




dataset_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize([66, 200], antialias=True),
])


#################################################################
# Data analysis for training                                    #
#################################################################


def main():

    D = 2  # Decimales (no se usa directamente aquí, pero podrías usarlo en labels si quieres)
    dataset = rosbagDataset(DATA_PATH, dataset_transforms)

    # Obtener velocidades lineales y angulares
    vels = [velocitys for image, velocitys in dataset.dataset]
    linear_velocities = np.array([vel[0] for vel in vels])
    angular_velocities = np.array([vel[1] for vel in vels])

    # Crear bordes para bins
    bins = 50  # Puedes ajustar esto
    linear_edges = np.linspace(linear_velocities.min(), linear_velocities.max(), bins)
    
    w_limit = max(abs(angular_velocities.min()), abs(angular_velocities.max()))
    angular_edges = np.linspace(-w_limit, w_limit, bins)

    # Histograma 2D
    hist, xedges, yedges = np.histogram2d(linear_velocities, angular_velocities, bins=(linear_edges, angular_edges))

    # Crear figura
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Posiciones de las barras
    xpos, ypos = np.meshgrid(
        xedges[:-1] + np.diff(xedges)[0] / 2,
        yedges[:-1] + np.diff(yedges)[0] / 2,
        indexing="ij"
    )
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    # Dimensiones de las barras
    dx = np.full_like(zpos, np.diff(xedges)[0] * 0.8)
    dy = np.full_like(zpos, np.diff(yedges)[0] * 0.8)
    dz = hist.ravel()

    # Colores por magnitud angular
    magnitude = np.abs(ypos)
    magnitude_norm = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude
    colors = np.zeros((len(dz), 4))
    colors[:, 0] = magnitude_norm  # rojo
    colors[:, 1] = 1 - magnitude_norm  # verde
    colors[:, 3] = 0.9  # opacidad

    # Dibujar barras
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')

    # Estética
    ax.set_xlabel('Velocidad Lineal (v)', labelpad=20, fontsize=15)
    ax.set_ylabel('Velocidad Angular (w)', labelpad=20, fontsize=15)
    ax.set_zlabel('Frecuencia', labelpad=20, fontsize=15)

    ax.view_init(elev=35, azim=135)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()



    # Metodos para desajuste:

    # 1. Oversampling
    # 2. Class weighting