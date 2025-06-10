#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
import os
from data_loader import *
from pylotnet_dataset import PilotNetDataset
from Reference.pilotnet import PilotNet
from Reference.transform_helper import createTransform

import argparse
from PIL import Image

import json
import numpy as np
import cv2
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


writer = SummaryWriter()

package_path = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control"
CSV_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/csvs/ACK"
CHECKPOINT_PATH = "/home/juan/ros2_tfg_ws/src/f1/dl_car_control/check_point"



def plot_velocity_histogram_3d(path_to_data, bins=50):
    
    linear_velocities = []
    angular_velocities = []

    for folder_path in path_to_data:
        file_path = os.path.join(folder_path, 'data.csv')
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Convierte valores a float, ignorando errores
                df['v'] = pd.to_numeric(df['v'], errors='coerce')
                df['w'] = pd.to_numeric(df['w'], errors='coerce')

                # Elimina filas con valores no numéricos
                df = df.dropna(subset=['v', 'w'])

                linear_velocities.extend(df['v'].values)
                angular_velocities.extend(df['w'].values)
            except Exception as e:
                print(f"[!] Error leyendo {file_path}: {e}")
        else:
            print(f"[!] No existe: {file_path}")

    if not linear_velocities or not angular_velocities:
        print("[!] No se encontraron datos válidos.")
        return

    linear_velocities = np.array(linear_velocities, dtype=float)
    angular_velocities = np.array(angular_velocities, dtype=float)


    # Rango simétrico para w
    w_limit = max(abs(angular_velocities.min()), abs(angular_velocities.max()))
    linear_bins = np.linspace(linear_velocities.min(), linear_velocities.max(), bins)
    angular_bins = np.linspace(-w_limit, w_limit, bins)

    hist, xedges, yedges = np.histogram2d(linear_velocities, angular_velocities,
                                          bins=[linear_bins, angular_bins])

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    xpos, ypos = np.meshgrid(
        xedges[:-1] + np.diff(xedges)[0] / 2,
        yedges[:-1] + np.diff(yedges)[0] / 2,
        indexing="ij"
    )
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    dx = np.diff(xedges)[0] * 0.8
    dy = np.diff(yedges)[0] * 0.8
    dz = hist.ravel()

    dx = np.full_like(zpos, dx)
    dy = np.full_like(zpos, dy)

    # Color por magnitud angular
    magnitude = np.abs(ypos)
    magnitude_norm = magnitude / (magnitude.max() if magnitude.max() != 0 else 1)

    colors = np.zeros((len(dz), 4))
    colors[:, 0] = magnitude_norm  # Rojo
    colors[:, 1] = 1 - magnitude_norm  # Verde
    colors[:, 3] = 0.9  # Opacidad

    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, zsort='average')

    # Estética mejorada
    ax.set_xlabel('Velocidad Lineal (v)', labelpad=20, fontsize=15)
    ax.set_ylabel('Velocidad Angular (w)', labelpad=20, fontsize=15)
    ax.set_zlabel('Frecuencia', labelpad=20, fontsize=15)

    ax.view_init(elev=35, azim=135)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.show()



if __name__=="__main__":


    # Base Directory
    path_to_data = []
    
    datasets = {
        "Montmelo_ACK": 7,
        "Montmelo_ACK_Dizzy": 2,
        "Montmelo_ACK_inv": 7,
        "Simple_ACK_inv": 3,
        "Simple_ACK_Dizzy": 1,
        "Hard_cases_ACK": 20,
    }

    for dataset, count in datasets.items():
        for _ in range(count):
            path_to_data.append(CSV_PATH + f'/{dataset}')

        
    
    path_to_data = path_to_data * 3
    
    plot_velocity_histogram_3d(path_to_data)
    exit(0)
    
    test_dir = [CSV_PATH + '/Simple_ACK']
    base_dir = package_path
    model_save_dir = CHECKPOINT_PATH
    log_dir = base_dir + '/log'

    # Hyperparameters
    augmentations = []#'all'
    num_epochs = 25
    batch_size = 128
    learning_rate = 1e-3
    val_split = 0.1
    shuffle_dataset = True
    save_iter = 50
    random_seed = 471
    print_terminal = True
    
    preprocess = 'crop'  #+ 'norm' + 'afine' 

    # Device Selection (CPU/GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FLOAT = torch.FloatTensor

    # Tensorboard Initialization
    writer = SummaryWriter(log_dir)

    # Define data transformations
    transformations = createTransform(augmentations)
    
    # transformations = transforms.Compose([
    #         transforms.ToTensor()
    #     ])
    # Load data
    dataset = PilotNetDataset(path_to_data, transformations, preprocess)
    
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_split = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(val_split)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    # Load Model
    pilotModel = PilotNet(dataset.image_shape, dataset.num_labels).to(device)
    if os.path.isfile( model_save_dir + '/pilot_net_model_{}.ckpt'.format(random_seed)):
        pilotModel.load_state_dict(torch.load(model_save_dir + '/pilot_net_model_{}.ckpt'.format(random_seed),map_location=device))
        last_epoch = json.load(open(model_save_dir+'/args.json',))['last_epoch']+1
    else:
        last_epoch = 0
    last_epoch = 0
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(pilotModel.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    global_iter = 0
    global_val_mse = 1e+5


    print("*********** Training Started ************")
    train_losses = []
    test_losses = []
    for epoch in range(last_epoch, num_epochs):
        pilotModel.train()
        running_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            
            # print("Velocidades ", labels[0].float())
            # img_array = []
            
            
            # for j in range(batch_size):
            #     imagenes= torch.permute(images[j], (1, 2, 0))
            #     img_array.append(imagenes.numpy())
            #     cv2.imshow("image", img_array[j])
            #     cv2.waitKey(0);  
            #     cv2.destroyAllWindows()
            #     print(img_array[j].shape)
            
      
            
            images = FLOAT(images).to(device)
            # img = images[0].cpu()
            labels = FLOAT(labels.float()).to(device)
            
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # Run the forward pass
            outputs = pilotModel(images)
            loss = criterion(outputs, labels)
            writer.add_scalar("Loss/train", loss, epoch+1)
            
            # Backprop and perform Adam optimisation
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            running_loss += current_loss

            if global_iter % save_iter == 0:
                torch.save(pilotModel.state_dict(), model_save_dir + '/pilot_net_model_{}.ckpt'.format(random_seed))
            global_iter += 1

            if print_terminal and (i + 1) % 10 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
        
        # add entry of last epoch                            
        with open(model_save_dir+'/args.json', 'w') as fp:
            json.dump({'last_epoch': epoch}, fp)


        # Validation 
        pilotModel.eval()
        with torch.no_grad():
            val_loss = 0 
            for images, labels in val_loader:
                images = FLOAT(images).to(device)
                labels = FLOAT(labels.float()).to(device)
                outputs = pilotModel(images)
                val_loss += criterion(outputs, labels).item()
                
            val_loss /= len(val_loader) # take average
            writer.add_scalar("performance/valid_loss", val_loss, epoch+1)
        
        # Registrar error de entrenamiento
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Registrar error de prueba
        test_losses.append(val_loss)

        # compare
        if val_loss < global_val_mse:
            global_val_mse = val_loss
            best_model = deepcopy(pilotModel)
            torch.save(best_model.state_dict(), model_save_dir + '/pilot_net_model_best_{}.pth'.format(random_seed))
            mssg = "Model Improved!!"
        else:
            mssg = "Not Improved!!"

        print('Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch + 1, num_epochs, val_loss), mssg)
        
    print('Finished Training')

    plt.plot(train_losses, label='Training loss')
    # plt.plot(test_losses, label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
                

    pilotModel = best_model # allot the best model on validation 
    # Test the model
    transformations_val = createTransform([]) # only need Normalize()
    test_set = PilotNetDataset(test_dir, transformations_val, preprocessing='crop')
    test_loader = DataLoader(test_set, batch_size=batch_size)
    print("Check performance on testset")
    pilotModel.eval()
    with torch.no_grad():
        test_loss = 0
        for images, labels in tqdm(test_loader):
            images = FLOAT(images).to(device)
            labels = FLOAT(labels.float()).to(device)
            outputs = pilotModel(images)
            test_loss += criterion(outputs, labels).item()
    
    writer.add_scalar('performance/Test_MSE', test_loss/len(test_loader))
    print(f'Test loss: {test_loss/len(test_loader)}')
        
    # Save the model and plot
    torch.save(pilotModel.state_dict(), model_save_dir + '/pilot_net_model_{}.ckpt'.format(random_seed))