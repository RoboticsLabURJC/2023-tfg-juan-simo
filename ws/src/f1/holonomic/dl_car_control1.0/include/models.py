#!/usr/bin/env python3

import torch
from torch import nn


class pilotNet(nn.Module):
    def __init__(self) -> None:
        super(pilotNet, self).__init__()

        self.network = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 24, 5, 2),
            nn.ReLU(),
            nn.Conv2d(24, 36, 5, 2),
            nn.ReLU(),
            nn.Conv2d(36, 48, 5, 2),
            nn.ReLU(),
            nn.Conv2d(48, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1152, 1164),
            nn.Linear(1164, 100),
            nn.Linear(100, 50),
            nn.Linear(50, 10),
            nn.Linear(10, 2),       # Segun el modelo tiene sentido pero sería así??? Se cambió a 2 ya que sería v angular y lineal en caso del coche
        )
        
    def forward(self, x):
        return self.network(x)