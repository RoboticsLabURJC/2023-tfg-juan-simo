#!/usr/bin/env python3

from rosbags.rosbag2 import Reader as ROS2Reader
from rosbags.serde import deserialize_cdr
import numpy as np
import cv2
import csv
import os
from data_loader import *
import matplotlib.pyplot as plt
from tqdm import tqdm

CSV_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/"
DEBUG = False

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def filter_red(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 70, 50])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    
    lower_red = np.array([170, 70, 50])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    
    mask = mask1 | mask2
    result = cv2.bitwise_and(image, image, mask=mask)
    
    if DEBUG:
        cv2.imshow('Filtered Red', result)
        cv2.waitKey(0)
        
    return result

def is_straight_line(contour):
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) <= 4:
        _, _, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        if aspect_ratio > 0.8 or aspect_ratio < 0.2:  # more lenient aspect ratio
            return True
    return False

def classify_line_shape(image):
    filtered_image = filter_red(image)
    gray = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)
    
    if DEBUG:
        cv2.imshow('Gray Image', gray)
        cv2.waitKey(0)
    
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    if DEBUG:
        cv2.imshow('Edges', edges)
        cv2.waitKey(0)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if DEBUG:
        debug_img = image.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 3)
        cv2.imshow('Contours', debug_img)
        cv2.waitKey(0)

    # Sort contours by length and keep only the two longest
    contours = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)[:2]

    line_count = 0
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Ignore small contours
            if is_straight_line(contour):
                line_count += 1
    
    # Adjusting the classification logic to be more lenient
    if line_count >= 1:
        return 'LINE'
    else:
        return 'CURVE'

def classify(path):
    line_path = os.path.join(path, "line2.2")
    curve_path = os.path.join(path, "curve2.2")
    
    check_path(line_path)
    check_path(curve_path)
    
    line_name = os.path.join(line_path, 'data.csv')
    curve_name = os.path.join(curve_path, 'data.csv')
    
    line_file = open(line_name, 'w', newline='')
    fieldnames = ['v', 'w']
    line_writer = csv.DictWriter(line_file, fieldnames=fieldnames)
    line_writer.writeheader()
    
    curve_file = open(curve_name, 'w', newline='')
    curve_writer = csv.DictWriter(curve_file, fieldnames=fieldnames)
    curve_writer.writeheader()
    
    labels = []
    all_images, all_data = load_data(path)
    labels = parse_csv(all_data, labels)
    
    for i in tqdm(range(len(labels))):
        img_name = os.path.basename(all_images[i])
        img = cv2.imread(all_images[i])
        
        shape = classify_line_shape(img)
        if DEBUG:
            print(shape)
        
        if shape == 'LINE':
            line_writer.writerow({'v': labels[i][0], 'w': labels[i][1]})
            img_path = os.path.join(line_path, img_name)
        else:
            curve_writer.writerow({'v': labels[i][0], 'w': labels[i][1]})
            img_path = os.path.join(curve_path, img_name)
        
        cv2.imwrite(img_path, img)
        
    line_file.close()
    curve_file.close()
    
def list_directories(directory):
    return [entry for entry in os.listdir(directory) if os.path.isdir(os.path.join(directory, entry))]

def main():
    
    if DEBUG:
        path = CSV_PATH + "Simple_E2"
        classify(path)
        return
    
    directories = list_directories(CSV_PATH)
    csv_paths = [os.path.join(CSV_PATH, dir) for dir in directories if "uncropped" not in dir and "Alex" not in dir]
    
    for path in csv_paths:
        classify(path)

if __name__ == "__main__":
    main()
