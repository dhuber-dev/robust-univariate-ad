import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


class FFModel(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(FFModel, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes

        # Define the architecture
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_shape[1], 512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(128, num_classes)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass for the model.

        :param x: torch.Tensor
            Input tensor of shape (batch_size, num_features).
        :return: torch.Tensor
            Output tensor of shape (batch_size, num_classes).
        """
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x
