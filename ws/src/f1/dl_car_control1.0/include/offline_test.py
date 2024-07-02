#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
import csv
import argparse
import cv2
import torch
from torchvision import transforms
from Reference.pilotnet import PilotNet
from datetime import datetime
from data_loader import *

TEST_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/Simple_E2"
MODEL_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/models/E2_2/pilot_net_model_121_afine.ckpt"

MAX_ANGULAR = 4.5 
MAX_LINEAR = 20 
MIN_LINEAR = 3

def denormalize(normalized_value, min, max):
    return (normalized_value * (max - min)) + min

def main():

    # Device Selection (CPU/GPU)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_type = "cuda"
    else:
        device = torch.device("cpu")
        device_type = "cpu"


    data_w_array= []
    net_w_array= []
    data_v_array= []
    net_v_array= []
    n_array = []
    count = 1
    

    # image_shape = (66, 200, 3)
    image_shape = [200, 66, 3]
    num_labels = 2
    input_size =[66, 200]

    pilotModel = PilotNet(image_shape, num_labels).to(device)
    pilotModel.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    pilotModel.eval()

    preprocess = transforms.Compose([
        # convert the frame to a CHW torch tensor for training
        transforms.ToTensor()
    ]) 
    
    images = []
    labels = []

    all_images, all_data = load_data(TEST_PATH)
    images = get_images(all_images, 'cropped', images)
    labels = parse_csv(all_data, labels)

    first_line = True
    total_time = 0
    min = 20000
    max = -1
    total_mult = 0
    total_loss_v = 0
    total_loss_w = 0
    norm = True
    
    for i in tqdm(range(len(images))):
        
        start_time = datetime.now()
        
        image = images[i]
        label = labels[i]
        
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # height, width
        height = image.shape[0]
        width = image.shape[1]
        
        # crop image
        # cropped_image = image[240:480, 0:640]

        resized_image = cv2.resize(image, (int(input_size[1]), int(input_size[0])))

        # Display cropped image
        # cv2.imshow("image", resized_image)       
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        input_tensor = preprocess(resized_image).to(device)

        # print(type(input_tensor))
        # The model can handle multiple images simultaneously so we need to add an
        # empty dimension for the batch.
        # [3, 200, 66] -> [1, 3, 200, 66]
        input_batch = input_tensor.unsqueeze(0)
        # Inference (min 20hz max 200hz)

        output = pilotModel(input_batch)
        
    
        

        if device_type == "cpu":
            v = output[0].detach().numpy()[0]
            w = output[0].detach().numpy()[1]

        else:         
            v = output.data.cpu().numpy()[0][0]
            w = output.data.cpu().numpy()[0][1]


        if norm:
            v = denormalize(v, MIN_LINEAR, MAX_LINEAR)
            w = denormalize(w, -MAX_ANGULAR, MAX_ANGULAR)

        net_v_array.append(v)
        net_w_array.append(w)

        total_loss_v = total_loss_v + abs(float(label[0])-v) 
        total_loss_w = total_loss_w + abs(float(label[1])-w)


        data_v_array.append(float(label[0]))
        data_w_array.append(float(label[1]))
        n_array.append(count)
        
        finish_time = datetime.now()
        dt = finish_time - start_time
        ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
        total_time = total_time + ms
        if ms < min:
            min = ms
        if ms > max:
            max = ms
        #if not float(line[2]) == 0:
            #total_mult = total_mult + (output[0].detach().numpy()[1]/float(line[2]))
        count = count + 1
        

    
    print("Tiempo medio:"+str(total_time/count))
    print("Error medio v:"+str(total_loss_v/count))
    print("Error medio w:"+str(total_loss_w/count))
    print("Tiempo min:"+str(min))
    print("Tiempo max:"+str(max))
    #print("Mult_w: "+ str(total_mult/count))

    
    plt.subplot(1, 2, 1)
    plt.plot(n_array, data_v_array, label = "controller", color='b')
    plt.plot(n_array, net_v_array, label = "net", color='tab:orange')
    plt.title("Lineal speed") 
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(n_array, data_w_array, label = "controller", color='b')
    plt.plot(n_array, net_w_array, label = "net", color='tab:orange')
    plt.title("Angular speed") 
    plt.legend()

    plt.show()
    
    print("FIN")



# Execute!
if __name__ == "__main__":
    main()