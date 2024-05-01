import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from .DataLoader_nxn import VideoActionDataset
from .Model import Simple3DCNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose(
        [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    )
    dataset = VideoActionDataset(
        logs_dir="path/to/logs",
        videos_dir="path/to/videos",
        num_frames=60,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = Simple3DCNN(num_classes=10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 2
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 10}")
                running_loss = 0.0

    print("Finished Training")
