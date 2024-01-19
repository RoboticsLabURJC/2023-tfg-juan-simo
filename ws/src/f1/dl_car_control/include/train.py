#!/usr/bin/env python3

from itertools import islice
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch
from torch.serialization import save
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import sys
from models import pilotNet
import keyboard
from data_selection import rosbagDataset, dataset_transforms, DATA_PATH
import ament_index_python
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

writer = SummaryWriter()

package_path = "/home/juan/ros2_ws/src/f1/dl_car_control"
CHECKPOINT_PATH = "/home/juan/ros2_ws/src/f1/dl_car_control/check_point/network.tar"


def should_resume():
    return "--resume" in sys.argv or "-r" in sys.argv


def save_checkpoint(model: pilotNet, optimizer: optim.Optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, CHECKPOINT_PATH)


def load_checkpoint(model: pilotNet, optimizer: optim.Optimizer = None):
    checkpoint = torch.load(CHECKPOINT_PATH)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer != None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])


def evaluate(model, test_loader, criterion):
    model.eval()  # Cambia el modelo a modo de evaluación
    total_loss = 0
    with torch.no_grad():  # No necesitamos calcular gradientes
        for data in test_loader:
            label, image = data
            outputs = model(image)
            loss = criterion(outputs, label)
            total_loss += loss.item()
    return total_loss / len(test_loader)



def train(model: pilotNet, optimizer: optim.Optimizer):

    criterion = nn.MSELoss()

    dataset = rosbagDataset(DATA_PATH, dataset_transforms)

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.25)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    # train_loader = DataLoader(dataset, batch_size=50, shuffle=True)

    train_losses = []
    test_losses = []

    for epoch in range(200):

        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):

            # get the inputs; data is a list of [inputs, labels]
            label, image = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(image)
            loss = criterion(outputs, label)
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / (2000)))
                running_loss = 0.0

                save_checkpoint(model, optimizer)
        
        # Registrar error de entrenamiento
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Registrar error de prueba
        test_loss = evaluate(model, test_loader, criterion)
        test_losses.append(test_loss)
                

    print('Finished Training')

    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    model = pilotNet()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0)

    if should_resume():
        load_checkpoint(model, optimizer)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )   
    print(f"Using {device} device")
    model.to(device)
    model.train(True)

    train(
        model,
        optimizer,
    )