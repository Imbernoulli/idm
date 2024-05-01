import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize

from ActionMapping import ACTION_NUM
from Loss import MixedLoss
from DataLoader_nxn import DataLoader_nxn
from Model import VideoActionModel


def main():
    batch_size = 32
    num_epochs = 2
    learning_rate = 0.0001

    # 数据转换
    transform = Compose(
        [
            transforms.ToPILImage(),
            ToTensor(),
            transforms.Resize((1280, 720)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 创建数据集实例
    dataset = DataLoader_nxn(
        logs_dir="/Users/bernoulli_hermes/projects/cad/detect/logs_refined",
        videos_dir="/Users/bernoulli_hermes/projects/cad/detect/videos",
        num_frames=60,
        transform=transform,
    )

    # 划分数据集
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 创建DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型、损失函数和优化器
    model = VideoActionModel(ACTION_NUM).to(device)
    criterion = MixedLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练循环
    for epoch in range(num_epochs):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        test_loss = evaluate(model, test_dataloader, criterion, device)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}"
        )

    # 保存训练好的模型
    torch.save(model.state_dict(), "trained_model.pth")


# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for frames, labels in dataloader:
        frames = frames.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


# 测试函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for frames, labels in dataloader:
            frames = frames.to(device)
            labels = labels.to(device)
            outputs = model(frames)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


if __name__ == "__main__":
    main()
