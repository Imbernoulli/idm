import torch
import torch.nn as nn


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.fc1 = nn.Linear(
            128 * 30 * 8 * 8, 512
        )  # Assuming the output from conv layers is 128 * 30 * 8 * 8
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        # Flatten the layers
        x = x.view(x.size(0), -1)  # Flatten for passing to fully connected layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
