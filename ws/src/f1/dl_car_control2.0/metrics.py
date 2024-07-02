#!/usr/bin/env python3

## Links used
# https://github.com/ros2/rosbag2
# https://pytorch.org/tutorials/beginner/introyt/trainingyt.html

import rclpy
from rclpy.node import Node
from rclpy.clock import ROSClock

from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped

import numpy as np
import cv2
from cv_bridge import CvBridge
import time

import matplotlib.pyplot as plt


MIN_PIXEL = -360
MAX_PIXEL = 360


# Red filter parameters
color_mask = ([17, 15, 70], [50, 56, 255])

white_c_mask = ([0, 0, 0], [0, 0, 0])

# Trace parameters
TRACE_COLOR = [0, 255, 0]
RADIUS = 2

# Distance to search the red line end
LIMIT_UMBRAL = 22

# Lap state
NOT_STARTED = 0
STARTED = 1
FINNISH_LINE = 2
FINNISHED = 3
STARTING = 4

import signal
import sys



# Assuming self.speeds and self.dists are already defined and contain the data
def plot_data(speeds, dists):
    plt.figure()

    # Plot speeds
    plt.subplot(2, 1, 1)
    plt.plot(speeds, label='Speed')
    plt.title('Speeds')
    plt.xlabel('Iteration')
    plt.ylabel('Speed')
    plt.legend()

    # Plot distances
    plt.subplot(2, 1, 2)
    plt.plot(dists, label='Distance')
    plt.title('Distances')
    plt.xlabel('Iteration')
    plt.ylabel('Distance')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()


class MetricsController(Node):

    def __init__(self):
        super().__init__('f1_line_follow')
        signal.signal(signal.SIGINT, self.signal_handler)

        self.filteredPublisher_ = self.create_publisher(Image, '/filtered_img', 100)

        self.imageSubscription = self.create_subscription(Image, '/cam_f1_left/image_raw', self.listener_callback, 10)

        self.velSubscription = self.create_subscription(Twist, '/cmd_vel', self.vel_callback, 10)

        self.px_rang = MAX_PIXEL - MIN_PIXEL


        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.filteredImage = None
        self.whiteImage = None
        self.receivedImage = None

        # Metrics
        self.dt = 0
        self.n_its = 0
        
        self.total_dist = 0
        self.max_dist = 0
        
        self.dists = []
        
        self.lap_state = 0
        self.lap_time = []
        self.start_time = 0
        self.best_lap = 1000
        
        self.max_speed = 0
        self.min_speed = 100
        self.total_speed = 0
        self.x_speed = 0
        self.speeds = []
        
        self.centered = 0
        
    def signal_handler(self, sig, frame):
        mean_dist = self.total_dist / self.n_its
        mean_speed = self.total_speed / self.n_its
        centered_percent = (self.centered/self.n_its) * 100
        total_time = 0
        n_laps = 0
        
        for lap in self.lap_time:
            total_time += lap
            n_laps += 1
        if n_laps < 0:
            n_laps = 1
            print("NOT 1 LAP FINISHED")
        mean_time = total_time / n_laps
        
        # Prepare data for the table
        table_data = [
            ["Mean dist to line", mean_dist],
            ["Max dist to line", self.max_dist],
            ["Mean lap time (s)", mean_time],
            ["Best lap time (s)", self.best_lap],
            ["Max speed", self.max_speed],
            ["Mean speed", mean_speed],
            ["Centered percent", centered_percent]
        ]

        # Create the figure
        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        # Plot speeds
        axs[0].plot(self.speeds, label='Speed')
        axs[0].set_title('Speeds')
        axs[0].set_xlabel('Iteration')
        axs[0].set_ylabel('Speed')
        axs[0].legend()

        # Plot distances
        axs[1].plot(self.dists, label='Distance')
        axs[1].set_title('Distances')
        axs[1].set_xlabel('Iteration')
        axs[1].set_ylabel('Distance')
        axs[1].legend()

        # Add table
        axs[2].axis('tight')
        axs[2].axis('off')
        table = axs[2].table(cellText=table_data, colLabels=["Metric", "Value"], cellLoc='center', loc='center')

        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.2, 1.2)

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

        sys.exit(0)


    def farest_point(self, image):
        img_width = image.shape[1]
        height_mid = int(image.shape[0] / 2)
            
        x = 0
        y = 0
        count = 0
            
        for row in range (height_mid, height_mid + LIMIT_UMBRAL):
            for col in range (img_width):
                    
                comparison = image[row][col] == np.array([0, 0, 0])
                if not comparison.all():
                    y += row
                    x += col 
                    count += 1
            
        start_row = height_mid
        end_row = height_mid + LIMIT_UMBRAL        
        # Add a green horizontal line at the first and last row evaluated
        if count > 0:
            image[start_row, :] = [0, 255, 0]  # Set the first row to green
            image[end_row - 1, :] = [0, 255, 0]  # Set the last row to green
            
        if (count == 0):
            return (0, 0)

        return [int(x / count), int(y / count)]

    def draw_traces(self, image):
        img_width = image.shape[1]
        img_height = image.shape[0]
            
        for row_index in range(img_height):
            image[row_index][int(img_width / 2)] = np.array(TRACE_COLOR)
            
    def start_line(self, image):
        
        img_width = int(image.shape[1]/4)
        img_height = image.shape[0]
        for row in range (img_height):
            for col in range (10, 20):
                    
                comparison = image[row][col] == np.array([0, 0, 0])
                if not comparison.all():
                    return True
        return False
    
    def vel_callback(self, msg):
        self.x_speed = msg.linear.x
        
    def listener_callback(self, msg):

        self.receivedImage = msg
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
            
            
        self.receivedImage = cv_image

        # Apply a red filter to the image
        red_lower = np.array(color_mask[0], dtype = "uint8")
        red_upper = np.array(color_mask[1], dtype = "uint8")
        red_mask = cv2.inRange(cv_image, red_lower, red_upper)
        self.filteredImage = cv2.bitwise_and(cv_image, cv_image, mask=red_mask)
        
        
        # Convertir la imagen de BGR a HSV
        # hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        # Filtros de color blanco
        white_lower = np.array([170, 170, 170], dtype="uint8")
        white_upper = np.array([180, 180, 180], dtype="uint8")
        white_mask = cv2.inRange(cv_image, white_lower, white_upper)
        self.whiteImage = cv2.bitwise_and(cv_image, cv_image, mask=white_mask)
    
    
    def timer_callback(self):
        if np.any(self.filteredImage):
            width_center = self.filteredImage.shape[1] / 2
            red_farest = self.farest_point(self.filteredImage)

            # Comentar pruebas en linea recta y robustez añadida al programa
            if red_farest[0] != 0 or red_farest[1] != 0:
                
                self.n_its = self.n_its + 1
                
                distance = (width_center - red_farest[0])
                
                if abs(distance) > abs(self.max_dist):
                    self.max_dist = distance
                
                self.total_dist = self.total_dist + distance
                self.dists.append(distance)
                
            
                is_start = self.start_line(self.whiteImage)
                if is_start and self.lap_state == NOT_STARTED:
                    self.lap_state = STARTING
                elif not is_start and self.lap_state == STARTING:
                    self.lap_state = STARTED
                    self.start_time = time.time()
                elif is_start  and self.lap_state == STARTED:
                    self.lap_state = FINNISH_LINE
                elif not is_start and self.lap_state == FINNISH_LINE:
                    end_time = time.time()
                    elapsed_time = end_time - self.start_time
                    print("LAP TIME = ", elapsed_time, " seconds")
                    if elapsed_time < self.best_lap:
                        self.best_lap = elapsed_time
                        
                    self.lap_time.append(elapsed_time)
                    self.lap_state = STARTED
                    self.start_time = time.time()

                
                self.total_speed = self.total_speed + self.x_speed
                self.speeds.append(self.x_speed)
                if self.x_speed > self.max_speed:
                    self.max_speed = self.x_speed
                # elif self.x_speed < self.min_speed:
                #     self.min_speed = self.x_speed
                
                comparison = self.filteredImage[479][390] == np.array([0, 0, 0])
                if not comparison.all():
                    self.centered += 1
                    
                if self.dt == 10: 
                    print("distance: ", distance, ": State = ", self.lap_state, "; Speed = ", self.x_speed)
                    self.dt = 0
                self.dt += 1

            # Traces and draw image
            # self.draw_traces(self.filteredImage)
            # cv2.circle(self.filteredImage, red_farest, RADIUS, TRACE_COLOR, RADIUS)

            # cv2.imshow("Filtered image", self.filteredImage)
            # cv2.imshow("White image", self.whiteImage)
            # cv2.imshow("Received image", self.receivedImage)
            cv2.waitKey(1)




def main(args=None):
    rclpy.init(args=args)
    
    minimal_publisher = MetricsController()

    rclpy.spin(minimal_publisher)

    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()