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



from subsets2 import classify_line_shape
from hard_cases import classify

DATA_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/ACK"

NURBU_MAP_IMG = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/Nurburing_circuit.png"
MONT_MAP_IMG = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/Montmelo_circuit.png"
SIMP_MAP_IMG = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/Simple_circuit.png"

CLASSIFIED_MAP = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/Simple_circuit_classified.png"
ROUTES_MAP = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/Simple_circuit_routed.png"
D_GRAPH_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/Distance_to_Line_Mont.png"
D_LOS_GRAPH_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/Distance_to_Line_Mont_Loss.png"

V_GRAPH_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/V_Simple.png"
W_GRAPH_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/W_Simple.png"

V_LOSS_GRAPH_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/V_Simple_Loss.png"
W_LOSS_GRAPH_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/car_ackerman/include/imgs/W_Simple_Loss.png"


COLOR = {
            'LINE': (255, 0, 0),   # Blue
            'CURVE': (0, 255, 0),  # Green
            'EXPERT': (255, 0, 0),  # Blue
            'MONOLITIC': (0, 255, 0),  # Green
            'COMBINATION': (0, 0, 255),  # Red
        }



# # Nurbu
# SCALE_X = 0.7466
# SCALE_Y = -0.7401
# WORLD_X0 = -633.28
# WORLD_Y0 = 281.34

# Monmelo
SCALE_X = -0.7195
SCALE_Y =  0.7395
WORLD_X0 = 605.07
WORLD_Y0 = -211.48

# # Simple
# SCALE_X =  0.7448
# SCALE_Y = -0.7339
# WORLD_X0 = -298.65
# WORLD_Y0 =  285.18

MARGIN = 2.5
first_pos = 0


# Function to convert world coordinates to pixel
def world_to_pixel(wx, wy):
    px = int((wx - WORLD_X0) / SCALE_X)
    py = int((wy - WORLD_Y0) / SCALE_Y)
    return px, py

def pixel_to_world(px, py):
    wx = px * SCALE_X + WORLD_X0
    wy = py * SCALE_Y + WORLD_Y0
    return wx, wy


# Load CSV
def load_positions(csv_path):
    positions = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = float(row['x'])
            y = float(row['y'])
            positions.append((x, y))
    print(f"Positions loaded from {csv_path}")
    return positions

def load_positions_in_time(csv_path, first=True):
    positions = []
    global first_pos

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # print(row)
            x = float(row['x'])
            y = float(row['y'])
            t = float(row['t'])
            if first:
                first = False
                first_pos = (x, y)
                
            positions.append((x, y, t))
    print(f"Positions loaded from {csv_path}")
    return positions

def add_legend(image, label, time_value=None, time_frame=None):

    # Define the legend's position and size
    legend_x = image.shape[1] - 175 
    legend_y = 10 

    # Draw the legend text without the white background
    if label == "CLASSIFY":
        cv2.putText(image, "Line (Blue)", (legend_x, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.putText(image, "Curve (Green)", (legend_x, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    if label == "TRACE":
        cv2.putText(image, "Expert (Blue)", (legend_x, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, "Monolitic (Green)", (legend_x, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image, "Combination (Red)", (legend_x, legend_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        if time_value is not None:
            cv2.putText(image, f"Time: {time_value:.1f}s", (legend_x, legend_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
        if time_frame is not None:
            cv2.putText(image, f"x{time_frame:.1f}", (legend_x + 100, legend_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)
            
    return image

def classify_route(image_path, positions, delay=0.01, extra_frames=120):
    map = cv2.imread(image_path)
    final_map = map.copy()

    if map is None:
        print(f"Error: failed to load image {image_path}")
        return

    imgs_path = DATA_PATH + "/Nurburing_ACK_ODOM"
    list_images = glob.glob(imgs_path + '/*.png')
    car_imgs = sorted(list_images, key=lambda x: int(x.split('/')[-1].split('.png')[0]))

    trail = []
    for i, (x, y, t) in enumerate(positions):
        if i >= len(positions) - extra_frames:  # Stop if i reaches max_frames (200)
            print(f"⏹️ Stopped at frame {i} as max_frames is reached.")
            break
        px, py = world_to_pixel(x, y)
        # Classify it
        car_img = cv2.imread(car_imgs[i])
        label = classify_line_shape(car_img)

        color = COLOR.get(label) 
        trail.append(((px, py), color))

        # Draw frame
        frame = map.copy()
        for pt, col in trail:
            cv2.circle(frame, pt, 7, col, -1)

        cv2.imshow("Route Playback", frame)
        key = cv2.waitKey(int(delay * 1000))
        if key == 27:  # ESC to exit
            break

        # Also draw permanently on the final image
        cv2.circle(final_map, (px, py), 7, color, -1)

    # Add the legend to the final map
    final_map = add_legend(final_map, "CLASSIFY")
    # Save final map
    cv2.imwrite(CLASSIFIED_MAP, final_map)
    print(f"✅ Final classified route saved to {CLASSIFIED_MAP}")
    cv2.destroyAllWindows()
    
def trace_route(image_path, all_positions, delay=0.02, extra_frames=0):
    map = cv2.imread(image_path)
    final_map = map.copy()
    global first_pos

    if map is None:
        print(f"Error: failed to load image {image_path}")
        return

    # Prepare trails for each label
    trails = []
    times_list = []

    min_time = float('inf')
    max_time = float('-inf')

    for label, positions in all_positions:
        color = COLOR.get(label, (0, 0, 255))  # Default to red if unknown
        trail = []
        times = []
        laps = -1
        laped = 0

        for i, (x, y, t) in enumerate(positions):
            if i >= len(positions) - extra_frames:
                break
            px, py = world_to_pixel(x, y)
            trail.append(((px, py), color))
            times.append(t)

            min_time = min(min_time, t)
            max_time = max(max_time, t)

            cv2.circle(final_map, (px, py), 1, color, -1)  # Draw on final image
            
            if (x <= first_pos[0] + MARGIN) and (x >= first_pos[0] - MARGIN) and (y <= first_pos[1] + MARGIN) and (y >= first_pos[1] - MARGIN) and (i > laped):
                laps += 1
                laped = i + 50
                if laps > 0:
                    print(f"LAP {laps} COMPLEATED")
                    break

        trails.append(trail)
        times_list.append(times)
         

    # Setup
    current_indices = [0] * len(trails)  # Current position index for each trail
    current_positions = [None] * len(trails)  # Current pixel position for each trail

    # Playback animation
    current_time = min_time
    end_time = max_time
    dt = 0.2
    first_time = True
    

    while current_time <= end_time:
        frame = map.copy()

        for idx, (trail, times) in enumerate(zip(trails, times_list)):
            while current_indices[idx] + 1 < len(times) and times[current_indices[idx] + 1] <= current_time:
                current_indices[idx] += 1

            if current_indices[idx] < len(trail):
                pt, col = trail[current_indices[idx]]
                current_positions[idx] = (pt, col)

        # Draw all current positions
        for pos in current_positions:
            if pos:
                pt, col = pos
                cv2.circle(frame, pt, 5, col, -1)

        frame = add_legend(frame, "TRACE", time_value=(current_time - min_time), time_frame=(dt/delay))
        cv2.imshow("Route Playback", frame)

        if cv2.waitKey(int(delay * 1000)) == 27:
            break
        if first_time:
            first_time = False
            time.sleep(1)

        current_time += dt
        
        if current_time >= end_time:
            time.sleep(1)

    final_map = add_legend(final_map, "TRACE")
    cv2.imwrite(ROUTES_MAP, final_map)
    print(f"✅ Final traced routes saved to {ROUTES_MAP}")
    cv2.destroyAllWindows()


def is_red(pixel):
    lower_bound = np.array([17, 15, 70])
    upper_bound = np.array([50, 56, 255])
    return np.all(pixel >= lower_bound) and np.all(pixel <= upper_bound)

def find_closest_red_pixel(img, start_x, start_y):
    height = len(img)
    width = len(img[0])
    visited = set()
    queue = deque()
    
    queue.append((start_x, start_y))
    visited.add((start_x, start_y))
    
    while queue:
        x, y = queue.popleft()
        
        # Check if current pixel is red
        if is_red(img[y][x]):
            return (x, y)
        
        # Explore neighbors
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in visited:
                queue.append((nx, ny))
                visited.add((nx, ny))
    
    return None

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def distance_to_line(image_path, positions, data_path, name="FILE"):
    map = cv2.imread(image_path)

    if map is None:
        print(f"Error: failed to load image {image_path}")
        return

    distances = []
    for i, (x, y, t) in enumerate(positions):
           
        px, py = world_to_pixel(x, y)   
        rx, ry = find_closest_red_pixel(map, px, py)
        wx, wy = pixel_to_world(rx, ry)
        distance = euclidean_distance(x, y, wx, wy)
        distances.append(distance)
        
    print(f"Max distance to line in {name} is {max(distances)}")
    print(f"Min distance to line in {name} is {min(distances)}")

    # Load the data from the CSV
    data = pd.read_csv(data_path)

    # Add the 'd' column with the distances
    data['d'] = distances

    # Save the updated data back to a new CSV
    data.to_csv(data_path, index=False)


# To test red filter
def keep_red_make_white(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image at {img_path}")
    
    height, width = img.shape[:2]
    
    new_img = np.ones_like(img) * 255  # Start with white image

    # Create a window
    cv2.namedWindow('Processing', cv2.WINDOW_NORMAL)

    for y in range(height):
        for x in range(width):
            pixel = img[y, x]
            if is_red(pixel):
                new_img[y, x] = pixel  # Keep red
            # else: already white
            
        # Optional: update every few rows for performance
        if y % 5 == 0:  
            cv2.imshow('Processing', new_img)
            cv2.waitKey(1)  # tiny pause to refresh window

    # Final display
    cv2.imshow('Processing', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return new_img

def moving_average(data, window_size=5):
    if window_size <= 0:
        raise ValueError("window_size must be greater than 0")
    if len(data) < window_size:
        raise ValueError("data length must be greater than or equal to window_size")

    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def plot_distances(csv_paths, downsample_factor=5, smooth=False, output_path=None):
    # Set up a plot
    plt.figure(figsize=(50, 15))  # Wider plot to fit multiple paths

    # Para almacenar distancias procesadas para ECM
    processed_data = {}  # label -> distances
    reference_label = csv_paths[0][1]  # Usamos el primer piloto como referencia
    for path, label in csv_paths:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(path)
        print(f"Getting {label} distances")
        
        # Extract the x, y, and distance (d) columns as numpy arrays
        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        distances = df['d'].to_numpy()

        # Calculate the cumulative distance along the track
        track_distances = np.zeros(len(x))
        laps = -1
        laped = 0
        last_i = 0
        for i in range(1, len(x)):
            track_distances[i] = track_distances[i-1] + np.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)

            if (x[i] <= first_pos[0] + MARGIN) and (x[i] >= first_pos[0] - MARGIN) and (y[i] <= first_pos[1] + MARGIN) and (y[i] >= first_pos[1] - MARGIN) and (i > laped):
                laps += 1
                laped = i + 50
                if laps > 0:
                    last_i = i
                    print(f"LAP {laps} COMPLEATED")
                    break
                
        x = x[:last_i]
        y = y[:last_i]
        distances = distances[:last_i]
        track_distances = track_distances[:last_i]

        x = x[::downsample_factor]
        y = y[::downsample_factor]
        distances = distances[::downsample_factor]
        track_distances = track_distances[::downsample_factor]

        if smooth:
            distances = moving_average(distances, window_size=5)
            track_distances = track_distances[:len(distances)]

        bgr_color = COLOR[label]
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        color = to_hex(np.array(rgb_color) / 255)

        plt.plot(track_distances, distances, label=label, color=color)

        # Guardar datos procesados para comparación
        processed_data[label] = distances

    # Add labels and title to the plot with increased fontsize
    plt.xlabel('Distance along Track (m)', fontsize=50)
    plt.ylabel('Mean Distance to Center of the Line (m)', fontsize=50)
    plt.title('Pilot Mean Distance to the Center of the Line', fontsize=50)
    plt.tick_params(axis='both', labelsize=30)
    plt.legend(fontsize=50)

    # Guardar o mostrar
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()

    # # === Cálculo del ECM respecto al piloto de referencia ===
    # print(f"\nResumen de ECM (Error Cuadrático Medio) respecto a '{reference_label}':\n")
    # ref_distances = processed_data[reference_label]
    # for label, dist in processed_data.items():
    #     if label == reference_label:
    #         continue
    #     min_len = min(len(ref_distances), len(dist))
    #     mse = mean_squared_error(ref_distances[:min_len], dist[:min_len])
    #     print(f"- {label}: {mse:.4f}")
    
    # # Nurbu:
    # - MONOLITIC: 0.9445
    # - COMBINATION: 0.8297
    # # Mont:
    # - MONOLITIC: 4.3976
    # - COMBINATION: 5.0656

    
    # === Cálculo del ECM respecto al centro (0) ===
    print("\nResumen de ECM (Error Cuadrático Medio respecto al centro de la pista):\n")

    for label, dist in processed_data.items():
        mse = mean_squared_error(np.zeros_like(dist), dist)
        print(f"- {label}: {mse:.4f}")
        
    # # Nurbu:
    # - EXPERT: 2.4653
    # - MONOLITIC: 2.9369
    # - COMBINATION: 2.7515
    # # Mont:
    # - EXPERT: 8.4559
    # - MONOLITIC: 8.2883
    # - COMBINATION: 8.2109
    # # Simple
    # - EXPERT: 5.0053
    # - MONOLITIC: 5.2449
    # - COMBINATION: 4.8693

def plot_speeds(csv_paths, downsample_factor=5, smooth=False, v_path=None, w_path=None, lv_path=V_LOSS_GRAPH_PATH, lw_path=W_LOSS_GRAPH_PATH):
    all_data = []
    
    for path, label in csv_paths:
        df = pd.read_csv(path)

        x = df['x'].to_numpy()
        y = df['y'].to_numpy()
        v = df['v'].to_numpy()
        w = df['w'].to_numpy()

        laps = -1
        laped = 0
        last_i = 0
        track_distances = np.zeros(len(x))
        for i in range(1, len(x)):
            track_distances[i] = track_distances[i-1] + np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)

            if (x[i] <= first_pos[0] + MARGIN) and (x[i] >= first_pos[0] - MARGIN) and \
               (y[i] <= first_pos[1] + MARGIN) and (y[i] >= first_pos[1] - MARGIN) and (i > laped):
                laps += 1
                laped = i + 50
                if laps > 0:
                    last_i = i
                    print(f"LAP {laps} COMPLETED for {label}")
                    break

        x = x[:last_i]
        y = y[:last_i]
        v = v[:last_i]
        w = w[:last_i]
        track_distances = track_distances[:last_i]

        # Downsample
        track_distances = track_distances[::downsample_factor]
        v = v[::downsample_factor]
        w = w[::downsample_factor]

        if smooth:
            v = moving_average(v, window_size=5)
            w = moving_average(w, window_size=5)
            track_distances = track_distances[:len(v)]

        all_data.append({
            'label': label,
            'track_distances': track_distances,
            'v': v,
            'w': w
        })

    # Plot velocidad lineal
    plt.figure(figsize=(50, 15))
    for data in all_data:
        bgr_color = COLOR[data['label']]
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        color = to_hex(np.array(rgb_color) / 255)

        plt.plot(data['track_distances'], data['v'], label=data['label'], color=color, linestyle='-')

    plt.xlabel('Distance along Track (m)', fontsize=50)
    plt.ylabel('Linear Velocity', fontsize=50)
    plt.title('Linear Velocity Along Track', fontsize=50)
    plt.tick_params(axis='both', labelsize=30)
    plt.legend(fontsize=30)

    if v_path:
        plt.savefig(v_path)
    else:
        plt.show()

    # Plot velocidad angular
    plt.figure(figsize=(50, 15))
    for data in all_data:
        bgr_color = COLOR[data['label']]
        rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
        color = to_hex(np.array(rgb_color) / 255)

        plt.plot(data['track_distances'], data['w'], label=data['label'], color=color, linestyle='--')

    plt.xlabel('Distance along Track (m)', fontsize=50)
    plt.ylabel('Angular Velocity', fontsize=50)
    plt.title('Angular Velocity Along Track', fontsize=50)
    plt.tick_params(axis='both', labelsize=30)
    plt.legend(fontsize=30)

    if w_path:
        plt.savefig(w_path)
    else:
        plt.show()

    # Calcular MSE global e imprimir
    ref = all_data[0]
    print(f"\nMSE respecto al primer piloto ({ref['label']}):")
    for data in all_data[1:]:
        min_len_v = min(len(ref['v']), len(data['v']))
        min_len_w = min(len(ref['w']), len(data['w']))

        mse_v = mean_squared_error(ref['v'][:min_len_v], data['v'][:min_len_v])
        mse_w = mean_squared_error(ref['w'][:min_len_w], data['w'][:min_len_w])
        print(f"  {data['label']} -> MSE Linear Velocity: {mse_v:.5f}, MSE Angular Velocity: {mse_w:.5f}")

    # # Graficar error cuadrático instantáneo para velocidad lineal (todos en la misma gráfica)
    # plt.figure(figsize=(20,6))
    # for data in all_data[1:]:
    #     min_len_v = min(len(ref['v']), len(data['v']))
    #     error_cuadratico_v = (data['v'][:min_len_v] - ref['v'][:min_len_v])**2
    #     bgr_color = COLOR[data['label']]
    #     rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
    #     color = to_hex(np.array(rgb_color) / 255)
    #     plt.plot(ref['track_distances'][:min_len_v], error_cuadratico_v,
    #              label=f'{data["label"]} vs {ref["label"]}', color=color)

    # plt.xlabel('Distance along Track (m)')
    # plt.ylabel('Error cuadrático instantáneo (velocidad lineal)')
    # plt.title('Error cuadrático instantáneo de la velocidad lineal')
    # plt.legend()
    # plt.savefig(lv_path)

    # # Graficar error cuadrático instantáneo para velocidad angular (todos en la misma gráfica)
    # plt.figure(figsize=(20,6))
    # for data in all_data[1:]:
    #     min_len_w = min(len(ref['w']), len(data['w']))
    #     error_cuadratico_w = (data['w'][:min_len_w] - ref['w'][:min_len_w])**2
    #     bgr_color = COLOR[data['label']]
    #     rgb_color = (bgr_color[2], bgr_color[1], bgr_color[0])
    #     color = to_hex(np.array(rgb_color) / 255)
    #     plt.plot(ref['track_distances'][:min_len_w], error_cuadratico_w,
    #              label=f'{data["label"]} vs {ref["label"]}', color=color)

    # plt.xlabel('Distance along Track (m)')
    # plt.ylabel('Error cuadrático instantáneo (velocidad angular)')
    # plt.title('Error cuadrático instantáneo de la velocidad angular')
    # plt.legend()
    # plt.savefig(lw_path)
    # # Nurbu
    # MONOLITIC -> MSE Linear Velocity: 14.07739, MSE Angular Velocity: 0.01057
    # COMBINATION -> MSE Linear Velocity: 16.10997, MSE Angular Velocity: 0.01120
    # # Mont
    # MONOLITIC -> MSE Linear Velocity: 13.77317, MSE Angular Velocity: 0.00823
    # COMBINATION -> MSE Linear Velocity: 13.39087, MSE Angular Velocity: 0.00763
    # # Simple
    # MONOLITIC -> MSE Linear Velocity: 10.22895, MSE Angular Velocity: 0.00660
    # COMBINATION -> MSE Linear Velocity: 12.53712, MSE Angular Velocity: 0.00699


def main(args=None):
    expert_csv_path = DATA_PATH + "/Montmelo_E_2.5/data_offset.csv"
    neural_csv_path = DATA_PATH + "/Montmelo_N_2.5/data_offset.csv"
    dneural_csv_path = DATA_PATH + "/Montmelo_DN_1.5/data_offset.csv"
    
    expert_positions = load_positions_in_time(expert_csv_path)
    neural_positions = load_positions_in_time(neural_csv_path, False)
    dneural_positions = load_positions_in_time(dneural_csv_path, False)

    # classify_route(NURBU_MAP_IMG, expert_positions)
    
    pilots_postitions = []
    pilots_postitions.append(("EXPERT", expert_positions))
    pilots_postitions.append(("MONOLITIC", neural_positions))
    pilots_postitions.append(("COMBINATION", dneural_positions))
    # trace_route(SIMP_MAP_IMG, pilots_postitions)
    
    # keep_red_make_white(NURBU_MAP_IMG)
    # distance_to_line(MONT_MAP_IMG, expert_positions, expert_csv_path, name="EXPERT")
    # distance_to_line(MONT_MAP_IMG, neural_positions, neural_csv_path, name="MONOLITIC")
    # distance_to_line(MONT_MAP_IMG, dneural_positions, dneural_csv_path, name="COMBINATION")
    
    csvs = []
    csvs.append((expert_csv_path, "EXPERT"))
    csvs.append((neural_csv_path, "MONOLITIC"))
    csvs.append((dneural_csv_path, "COMBINATION"))
    plot_distances(csvs, downsample_factor=5, smooth=True, output_path=D_GRAPH_PATH)
    # plot_speeds(csvs, downsample_factor=10, smooth=True, v_path=V_GRAPH_PATH, w_path=W_GRAPH_PATH)

    
if __name__ == '__main__':
    main()