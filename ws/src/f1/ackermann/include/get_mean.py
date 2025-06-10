#!/usr/bin/env python3

import csv
import time
import cv2
import numpy as np
import glob
from collections import deque
import math
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d




from subsets2 import classify_line_shape
from hard_cases import classify

DATA_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/ACK"
MARGIN = 1.5

first_pos = 0

def load_all(csv_path, first=True):
    positions = []
    global first_pos

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # print(row)
            v = float(row['v'])
            w = float(row['w'])
            x = float(row['x'])
            y = float(row['y'])
            t = float(row['t'])
            if first:
                first = False
                first_pos = (x, y)
                
            positions.append((v, w, x, y, t))
    return positions


def interpolate_lap(lap, target_length):
    lap = np.array(lap)
    original_indices = np.linspace(0, 1, len(lap))
    target_indices = np.linspace(0, 1, target_length)
    
    # Separar cada variable (v, w, x, y, t)
    interpolated = []
    for col in range(lap.shape[1]):
        f = interp1d(original_indices, lap[:, col], kind='linear')
        interpolated.append(f(target_indices))
    
    return np.stack(interpolated, axis=1)

def get_all_means(expert_data, data_output):
    global first_pos
    n_laps = -1
    i_laped = 0
    t_start = 0
    times = []
    separation = []
    for i, (v, w, x, y, t) in enumerate(expert_data):
        
        if (x <= first_pos[0] + MARGIN) and (x >= first_pos[0] - MARGIN) and (y <= first_pos[1] + MARGIN) and (y >= first_pos[1] - MARGIN) and (i > i_laped):
            n_laps += 1
            i_laped = i + 50
            if n_laps > 0:
                t_lap = t - t_start
                print(f"LAP {n_laps} COMPLEATED IN {t_lap}")
                times.append(t_lap)
                t_start = t
                separation.append(i)
                
    print(times)
    mean_time = 0
    for i, time in enumerate(times):
        mean_time += time
    mean_time = mean_time/(i+1)
    print(mean_time)
                
    laps = []
    for i in range(len(separation)):
        if i == 0:
            start_data = 0
        else:
            start_data = separation[i-1]
        end_data = separation[i]
        lap = expert_data[start_data:end_data]
        # print(len(lap))
        laps.append(lap)
        
    target_length = int(np.mean([len(lap) for lap in laps]))
    interpolated_laps = [interpolate_lap(lap, target_length) for lap in laps]
    mean_lap = np.mean(np.stack(interpolated_laps), axis=0)
    
    # Dynamically normalize the 't' column
    headers = ["v", "w", "x", "y", "t"]
    t_index = headers.index("t")
    mean_lap[:, t_index] -= mean_lap[0, t_index]
    
    # # Save to CSV
    # with open(data_output, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(headers)  # Column headers
    #     writer.writerows(mean_lap)

    print(f"Saved mean_lap to {data_output}")
    

def main(args=None):
    expert_csv_path = DATA_PATH + "/Nurbu_E_10/data.csv"
    neural_csv_path = DATA_PATH + "/Nurbu_N_10/data.csv"
    dneural_csv_path = DATA_PATH + "/Nurbu_DN_10/data.csv"
    
    expert_data = load_all(expert_csv_path)
    neural_data = load_all(neural_csv_path, False)
    dneural_data = load_all(dneural_csv_path, False)
    
    get_all_means(expert_data, expert_csv_path)
    get_all_means(neural_data, neural_csv_path)
    get_all_means(dneural_data, dneural_csv_path)
    
# LAP 1 COMPLEATED IN 164.213
# LAP 2 COMPLEATED IN 159.881
# LAP 3 COMPLEATED IN 162.68400000000003
# [164.213, 159.881, 162.68400000000003]
# 162.25933333333333
# Saved mean_lap to /home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/ACK/Simple_E_10/data.csv
# LAP 1 COMPLEATED IN 176.519
# LAP 2 COMPLEATED IN 175.735
# LAP 3 COMPLEATED IN 173.94900000000007
# [176.519, 175.735, 173.94900000000007]
# 175.40100000000004
# Saved mean_lap to /home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/ACK/Simple_N_10/data.csv
# LAP 1 COMPLEATED IN 170.535
# LAP 2 COMPLEATED IN 170.24599999999995
# [170.535, 170.24599999999995]
# 170.39049999999997
# Saved mean_lap to /home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/ACK/Simple_DN_10/data.csv

    
if __name__ == '__main__':
    main()