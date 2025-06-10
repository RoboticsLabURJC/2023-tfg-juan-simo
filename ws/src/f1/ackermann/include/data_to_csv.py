
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
import argparse


import matplotlib.pyplot as plt



DATA_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/rosbagsCar/"
CSV_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/ACK/"

IMG_TOPIC = "/car_img"
VEL_TOPIC = "/f1ros2/cmd_vel"
ODOM_TOPIC = "/car_odom"
CLOCK_TOPIC = "/car_clock"
DIZY_VEL_TOPIC = "/car_vel"



def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")
        return False
    return True


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
                    
                    # Cut the superior (upper) half of the image
                    half_height = resizeImg.shape[0] // 2 + 9 # for some reason this ackerman tracks have the image a bit lowered
                    bottom_half = resizeImg[half_height:, :, :]
                    
                    # # Mostrar la imagen
                    # cv2.imshow('Imagen', resizeImg)
                    # # Esperar a que el usuario presione una tecla para cerrar la ventana
                    # cv2.waitKey(0)

                    # Save the image
                    img_path = storage_dir + '/' + str(img_name) + '.png'
                    cv2.imwrite(img_path, cv2.cvtColor(bottom_half, cv2.COLOR_RGB2BGR))
                    
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



def get_pos(rosbag_dir, topic, storage_dir):
    name_file = storage_dir + '/data.csv'
    
    
    with open(name_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        
    poses = []
    
    with ROS2Reader(rosbag_dir) as ros2_reader:
        ros2_conns = [x for x in ros2_reader.connections]
        ros2_messages = ros2_reader.messages(connections=ros2_conns)
            
        for m, msg in enumerate(ros2_messages):
            (connection, timestamp, rawdata) = msg
            if (connection.topic == topic):
                data = deserialize_cdr(rawdata, connection.msgtype)

                # Extraer posición
                x = data.pose.pose.position.x
                y = data.pose.pose.position.y
                poses.append((x, y))
                
    if rows:
        for i in range(min(len(rows), len(poses))):
            rows[i]['x'] = poses[i][0]
            rows[i]['y'] = poses[i][1]
    else:
        rows = [{'x': x, 'y': y} for x, y in poses]

    fieldnames = list(rows[0].keys()) if rows else ['t']

    with open(name_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

       
                    
def get_time(rosbag_dir, topic, storage_dir):
    name_file = storage_dir + '/data.csv'
    
    with open(name_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        
    times = []

    with ROS2Reader(rosbag_dir) as ros2_reader:
        ros2_conns = [x for x in ros2_reader.connections]
        ros2_messages = ros2_reader.messages(connections=ros2_conns)
        
        for m, msg in enumerate(ros2_messages):
            (connection, timestamp, rawdata) = msg
            if (connection.topic == topic):
                data = deserialize_cdr(rawdata, connection.msgtype)

                secs = data.clock.sec
                nanosecs = data.clock.nanosec
                time = secs + (nanosecs / 1000000000)
                times.append(time)

    if rows:
        for i in range(min(len(rows), len(times))):
            rows[i]['t'] = times[i]
    else:
        rows = [{'t': t} for t in times]

    fieldnames = list(rows[0].keys()) if rows else ['t']

    with open(name_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
def adjust_time_column(input_file, output_file=None):
    name_file = input_file + '/data.csv'
    # If no output file is specified, overwrite the input file
    if output_file is None:
        output_file = name_file

    with open(name_file, mode='r', newline='') as csvfile:
        reader = list(csv.reader(csvfile))
        
        if not reader:
            print(f"No data found in {name_file}")
            return
        
        header = reader[0]
        try:
            time_index = header.index('t')
        except ValueError:
            print(f"No column labeled 't' found in {name_file}")
            return
        
        # Get the starting time from the third column
        start_time = float(reader[1][time_index])

        # Adjust all times
        adjusted_rows = []
        for row in reader:
            try:
                time_value = float(row[time_index])
                row[time_index] = str(time_value - start_time)
            except (IndexError, ValueError):
                pass  # Keep the row unchanged if there's an issue
            adjusted_rows.append(row)

    # Write back the adjusted data
    with open(output_file, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(adjusted_rows)

    print(f"File '{name_file}' adjusted successfully!")


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
    # print(directories)
    
    parser = argparse.ArgumentParser(description='Procesar rosbag para extraer datos de odometría.')
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-v', '--vel', action='store_true', help='Extraer velocidades lineales y angulares')
    group.add_argument('-o', '--odom', action='store_true', help='Extraer posición x e y')

    args = parser.parse_args()
    
    for dir in directories:
        csv_path = CSV_PATH + dir
        data_path = DATA_PATH + "/" + dir
        if "Dizy" in dir:
            print("Dizy")
            vel_topic = DIZY_VEL_TOPIC
        else:
            vel_topic = VEL_TOPIC
            
        # print(csv_path)
        # print(data_path)
          
        # Create non existing paths and avoids re-processing existing ones
        # Comment and change indentation if you want to process and overwritte everything in the path
        if not check_path(csv_path):
            # get_img(data_path, IMG_TOPIC, csv_path)
              
            if args.vel:
                get_vel(data_path, vel_topic, csv_path)
            elif args.odom:
                get_vel(data_path, vel_topic, csv_path)
                get_pos(data_path, ODOM_TOPIC, csv_path)
                get_time(data_path, CLOCK_TOPIC, csv_path)
                adjust_time_column(csv_path)
                
                
                

if __name__ == "__main__":
    main()


    # Metodos para desajuste:

    # 1. Oversampling
    # 2. Class weighting