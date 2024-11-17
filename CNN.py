import torch
import torch.nn as nn

class TimeSeriesCNN(nn.Module):
    def __init__(self, num_classes):
        super(TimeSeriesCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=2)

        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 124, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = torch.relu(x)
        x = self.pool(x)

        # Flatten the tensor for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x  # Return raw logits
