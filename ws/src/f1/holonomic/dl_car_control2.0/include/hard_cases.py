import numpy as np
import cv2
import csv
import os

from data_loader import *

CSV_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/"
HARD_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/Hard_cases_NODIZZY"

DEBUG = False

img_name = 0

def check_path(path):
    if not os.path.exists(path):
        print(f"{path} not exist")
        os.makedirs(path)
        print(f"Create {path} success")

def list_directories(directory):
    directories = []
    for entry in os.listdir(directory):
        full_path = os.path.join(directory, entry)
        if os.path.isdir(full_path):
            directories.append(entry)
    return directories

def classify(image):
    img_width = image.shape[1]  # 640
    img_height = image.shape[0] # 240
    image_copy = image.copy()
    
    
    if DEBUG:
        for colum in range(0, img_width-1):
            image_copy[5][colum] = np.array([0, 255, 0])
                
        # Mostrar la imagen
        cv2.imshow('Imagen', image_copy)
        # Esperar a que el usuario presione una tecla para cerrar la ventana
        cv2.waitKey(0)
    
    red = 0
    for colum in range(0, img_width-1):    
        comparison = image[0][colum] == np.array([0, 0, 0])
        if not comparison.all():
                red = 1
                break
    if red == 0:
        return True
    
    red = 0
    for row in range(0, 5):
        for colum in range(0, img_width-1):
            comparison = image[row][colum] == np.array([0, 0, 0])
            if not comparison.all():
                red = 1
                break
    if red == 0:
        return True
    
    if DEBUG:
        for row in range(0, img_height-1):
            image_copy[row][img_width - 250] = np.array([0, 255, 0])
                
        # Mostrar la imagen
        cv2.imshow('Imagen', image_copy)
        # Esperar a que el usuario presione una tecla para cerrar la ventana
        cv2.waitKey(0)
    
    red = 0 
    for colum in range(0, img_width - 250):
        for row in range(80, img_height-1):
            comparison = image[row][colum] == np.array([0, 0, 0])
            if not comparison.all():
                red = 1
                break
    if red == 0:
        return True
            
    if DEBUG:
        for row in range(0, img_height-1):
            image_copy[row][250] = np.array([0, 0, 255])
                
        # Mostrar la imagen
        cv2.imshow('Imagen', image_copy)
        # Esperar a que el usuario presione una tecla para cerrar la ventana
        cv2.waitKey(0)
    
    red = 0     
    for colum in range(250, img_width-1):
        for row in range(80, img_height-1):
            comparison = image[row][colum] == np.array([0, 0, 0])
            if not comparison.all():
                red = 1
                break
    if red == 0:
        return True
    
    
    return False
        
img_name = 0
def select(path):
    check_path(HARD_PATH)
    csv_name = HARD_PATH + '/data.csv'
    
    print(path)
    # Red filter parameters
    color_mask = ([17, 15, 70], [50, 56, 255])
    
    global img_name
    with open(csv_name, 'a', newline='') as hard_file:
        fieldnames = ['v', 'w']
        hard_writer = csv.DictWriter(hard_file, fieldnames=fieldnames)
        # Write the header
        hard_writer.writeheader()
        
        labels = []
        all_images, all_data = load_data(path)
        labels = parse_csv(all_data, labels)
        
        for i in tqdm(range(len(labels))):
                
            img = cv2.imread(all_images[i])
            
            # Apply a red filter to the image
            red_lower = np.array(color_mask[0], dtype = "uint8")
            red_upper = np.array(color_mask[1], dtype = "uint8")
            red_mask = cv2.inRange(img, red_lower, red_upper)
            filteredImage = cv2.bitwise_and(img, img, mask=red_mask)
            
            hard = classify(filteredImage)
            if hard:
                if DEBUG:
                    print("HARD")
                name = "HARD"
                hard_writer.writerow({'v': labels[i][0], 'w': labels[i][1]})
                
                # # Mostrar la imagen
                # cv2.imshow('Imagen', img)
                # # Esperar a que el usuario presione una tecla para cerrar la ventana
                # cv2.waitKey(0)
                # print(img_name)
                
                img_path = HARD_PATH + '/' + str(img_name) + '.png'
                cv2.imwrite(img_path, img)
                
                img_name += 1


def main():
    
    directories = list_directories(CSV_PATH)
    print(directories)
    csv_paths = []
    
    for dir in directories:
        if "uncropped" in dir:
            print("Skipping ", dir)
            continue
        if "Hard" in dir:
            continue
        if "Alex" in dir:
            print("Skipping ", dir)
            continue
        if "Dizy" in dir:
            print("Skipping ", dir)
            continue
        csv_paths.append(CSV_PATH + dir)
    
    print(csv_paths)   
    
    for path in csv_paths:
        select(path)


if __name__ == "__main__":
    main()
