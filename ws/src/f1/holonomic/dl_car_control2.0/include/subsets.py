#!/usr/bin/env python3


from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
import numpy as np
import cv2
import csv
import os
from data_loader import *

import matplotlib.pyplot as plt



CSV_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/"


ANGULAR_CRITERIA = 0.2
CENTER = 150
     

#################################################################
# Data analysis for training                                    #
#################################################################

def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")
        
def draw_traces(image):
        img_width = image.shape[1]  # 640
        img_height = image.shape[0] # 240
        
        height_mid = int(img_height / 2)
        width_mid = int(img_width / 2)
        count = 0
        y = 0
                
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
        #     #     image[colum][int((img_width / 2) + 100)] = np.array([0, 255, 0])
        #     #     image[colum][int((img_width / 2) - 100)] = np.array([0, 255, 0])
        #     # else:
        #     #     image[colum][int((img_width / 2) + 30)] = np.array([0, 255, 0])
        #     #     image[colum][int((img_width / 2) - 30)] = np.array([0, 255, 0])
                
        #     image[colum][mid] = np.array([0, 255, 0])
        #     image[colum][mid - 59] = np.array([0, 255, 0])
        #     image[colum][mid + 59] = np.array([0, 255, 0])
          
            
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
            
            
        


def classify(path):
    
    line_path = path + "/line"
    curve_path = path + "/curve"

    
    check_path(line_path)
    check_path(curve_path)
    
    line_name = line_path + '/data.csv'
    curve_name = curve_path + '/data.csv'
    
    # Red filter parameters
    color_mask = ([17, 15, 70], [50, 56, 255])
    
    line_file = open(line_name, 'w', newline='')
    fieldnames = ['v', 'w']
    line_writer = csv.DictWriter(line_file, fieldnames=fieldnames)
    # Write the header
    line_writer.writeheader()
    
    curve_file = open(curve_name, 'w', newline='')
    fieldnames = ['v', 'w']
    curve_writer = csv.DictWriter(curve_file, fieldnames=fieldnames)
    # Write the header
    curve_writer.writeheader()
    
    
    labels = []
    all_images, all_data = load_data(path)
    labels = parse_csv(all_data, labels)
    
    for i in tqdm(range(len(labels))):
        img_name = os.path.basename(all_images[i])
            
        img = cv2.imread(all_images[i])
        
        # Apply a red filter to the image
        red_lower = np.array(color_mask[0], dtype = "uint8")
        red_upper = np.array(color_mask[1], dtype = "uint8")
        red_mask = cv2.inRange(img, red_lower, red_upper)
        filteredImage = cv2.bitwise_and(img, img, mask=red_mask)
        
        line = draw_traces(filteredImage)
        
        if line:
            name = "LINE"
            line_writer.writerow({'v': labels[i][0], 'w': labels[i][1]})
            
            # # Mostrar la imagen
            # cv2.imshow('Imagen', img)
            # # Esperar a que el usuario presione una tecla para cerrar la ventana
            # cv2.waitKey(0)
            # print(img_name)
            
            img_path = line_path + '/' + img_name
            cv2.imwrite(img_path, img)
        else:
            name = "CURVE"
            curve_writer.writerow({'v': labels[i][0], 'w': labels[i][1]})             

            # # Mostrar la imagen
            # cv2.imshow('Imagen', img)
            # # Esperar a que el usuario presione una tecla para cerrar la ventana
            # cv2.waitKey(0)
            # print(img_name)
            
            img_path = curve_path + '/' + img_name
            cv2.imwrite(img_path, img)
        
    line_file.close()
    curve_file.close()
    
    
def list_directories(directory):
    directories = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            directories.append(entry)
    return directories


def main():
    
    directories = list_directories(CSV_PATH)
    print(directories)
    csv_paths = []
    
    for dir in directories:
        if "uncropped" in dir:
            print("Skipping ", dir)
            continue
        if "Alex" in dir:
            print("Skipping ", dir)
            continue
        csv_paths.append(CSV_PATH + dir)
    
    print(csv_paths)   
    
    for path in csv_paths:
        classify(path)
     
    

if __name__ == "__main__":
    main()


    # Metodos para desajuste:

    # 1. Oversampling
    # 2. Class weighting