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


def evaluate_FFModel(X_train, y_train, X_test, y_test):
    # hyperparams
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 2

    ## To tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FFModel(input_shape=X_train_tensor.shape, num_classes=len(set(np.concatenate((y_train, y_test)))))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        for i, (X_batch, y_batch) in enumerate(train_loader):
            ## Forward Pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            ## Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'epoch {epoch+1}/{num_epochs}, loss {loss.item()}:.2f')

    # Test the model
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_elements = len(test_loader.dataset)
    num_batches = len(test_loader)
    batch_size = test_loader.batch_size
    y_pred = torch.zeros(num_elements)
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            start = i * batch_size
            end = start + batch_size
            if i == num_batches - 1:
                end = num_elements
            y_pred[start:end] = predicted
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

    test_accuracy = correct / total
    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Accuracy: {test_accuracy:.4f}")

    return y_pred
