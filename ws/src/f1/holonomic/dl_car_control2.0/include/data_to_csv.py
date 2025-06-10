
#!/usr/bin/env python3

# run with python filename.py -i rosbag_dir/
# "../rosbagsCar/rosbag2_2023_10_09-11_50_46"
## Links: https://stackoverflow.com/questions/73420147/how-to-read-custom-message-type-using-ros2bag

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
import numpy as np
import cv2
import csv
import os


import matplotlib.pyplot as plt



DATA_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/rosbagsCar"
CSV_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/"

IMG_TOPIC = "/car_img"
VEL_TOPIC = "/cmd_vel"
DIZY_VEL_TOPIC = "/car_vel"



def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")


def get_img(rosbag_dir, topic, storage_dir):
    
        img_name = 0
        
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
                    
                    # # Cut the superior (upper) half of the image
                    # half_height = resizeImg.shape[0] // 2
                    # bottom_half = resizeImg[half_height:, :, :]
                    
                    # # Mostrar la imagen
                    # cv2.imshow('Imagen', resizeImg)
                    # # Esperar a que el usuario presione una tecla para cerrar la ventana
                    # cv2.waitKey(0)

                    # Save the image
                    img_path = storage_dir + '/' + str(img_name) + '.png'
                    cv2.imwrite(img_path, cv2.cvtColor(resizeImg, cv2.COLOR_RGB2BGR))
                    
                    img_name += 1
        
                    
def get_vel(rosbag_dir, topic, storage_dir):
        name_file = storage_dir + '/data.csv'
        
        with open(name_file, 'w', newline='') as file:
            fieldnames = ['v', 'w']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            # Write the header
            writer.writeheader()

            with ROS2Reader(rosbag_dir) as ros2_reader:

                ros2_conns = [x for x in ros2_reader.connections]
                ros2_messages = ros2_reader.messages(connections=ros2_conns)
                
                for m, msg in enumerate(ros2_messages):
                    (connection, timestamp, rawdata) = msg
                        
                    if (connection.topic == topic):
                        data = deserialize_cdr(rawdata, connection.msgtype)

                        linear = data.linear.x
                        angular = data.angular.z
                        # Write the data row as a dictionary
                        writer.writerow({'v': linear, 'w': angular})

                    

#################################################################
# Data analysis for training                                    #
#################################################################

def list_directories(directory):
    directories = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            directories.append(entry)
    return directories


def main():
    
    directories = list_directories(DATA_PATH)
    print(directories)
    
    for dir in directories:
        csv_path = CSV_PATH + dir
        data_path = DATA_PATH + "/" + dir
        if "Dizy" in dir:
            print("Dizy")
            vel_topic = DIZY_VEL_TOPIC
        else:
            vel_topic = VEL_TOPIC
        print(csv_path)
        print(data_path)
          
        check_path(csv_path)
        get_vel(data_path, vel_topic, csv_path)
        get_img(data_path, IMG_TOPIC, csv_path)
    

if __name__ == "__main__":
    main()


    # Metodos para desajuste:

    # 1. Oversampling
    # 2. Class weighting