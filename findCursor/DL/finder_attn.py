import json
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
from torchvision import transforms
import numpy as np
import pandas as pd
from torchvision.transforms import ToTensor
from torchvision.models import vit_b_16


class MouseMoveDataset(Dataset):
    def __init__(
        self,
        base_frames_folder,
        base_positions_folder,
        transform=ToTensor(),
        device=torch.device("cpu"),
    ):
        self.base_frames_folder = base_frames_folder
        self.base_positions_folder = base_positions_folder
        self.transform = transform
        self.device = device

        self.data = []

        # 遍历视频帧文件夹
        sessions = [
            d[:-7]
            for d in os.listdir(base_frames_folder)
            if os.path.isdir(os.path.join(base_frames_folder, d))
        ]

        for session in sessions:
            frames_folder = os.path.join(base_frames_folder, f"{session}_frames")
            position_file = os.path.join(
                base_positions_folder, f"{session}_positions.csv"
            )

            if os.path.exists(position_file):
                positions = pd.read_csv(
                    position_file, header=None, names=["frame_index", "x", "y"]
                )
                frame_files = {
                    int(f.split(".")[0]): os.path.join(frames_folder, f)
                    for f in os.listdir(frames_folder)
                    if f.endswith(".png")
                }

                for index, row in positions.iterrows():
                    frame_path = frame_files.get(row["frame_index"])
                    if frame_path:
                        self.data.append((frame_path, (row["x"], row["y"])))

    def __getitem__(self, idx):
        frame_path, (mouse_x, mouse_y) = self.data[idx]
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.transform:
            frame = self.transform(frame)

        frame = frame.to(self.device)  # Move frame to specified device

        mouse_position = torch.tensor([mouse_x, mouse_y], dtype=torch.float32).to(
            self.device
        )

        return frame, mouse_position

    def __len__(self):
        return len(self.data)


class MousePositionPredictor(nn.Module):
    def __init__(self):
        super(MousePositionPredictor, self).__init__()
        # 加载预训练的Vision Transformer模型，并指定image_size为224
        self.vit = vit_b_16(
            pretrained=True, weights=ViT_B_16_Weights.DEFAULT, image_size=224
        )
        # 假设ViT模型有768维输出
        self.fc1 = nn.Linear(768, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.vit(x)  # ViT模型处理
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for i, (images, mouse_positions) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, mouse_positions)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        print(f"Batch {i+1}/{total_batches}, Batch Loss: {loss.item():.4f}")

    average_loss = running_loss / len(dataloader.dataset)
    print(f"Training Loss: {average_loss:.4f}")
    return average_loss


# 测试函数
def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, mouse_positions in dataloader:
            outputs = model(images)
            loss = criterion(outputs, mouse_positions)

            running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


# 主函数
def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "4"

    batch_size = 64
    num_epochs = 5
    learning_rate = 0.0001
    train_val_split_ratio = 0.95  # 训练和验证集的比例
    save_path = "model.pth"

    model = MousePositionPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    # 设置设备
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model.to(device)

    # 创建数据集和数据加载器
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # 将图像大小调整为224x224
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # ImageNet预训练的标准化参数
        ]
    )

    dataset = MouseMoveDataset(
        base_frames_folder="frames",
        base_positions_folder="positions",
        transform=transform,
        device=torch.device("cuda"),
    )
    dataset_size = len(dataset)
    train_size = int(train_val_split_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_loss = float("inf")
    best_model_state_dict = None

    # 训练和验证模型
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = test(model, val_loader, criterion, device)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state_dict = model.state_dict()

    # 保存最佳模型参数
    torch.save(best_model_state_dict, f"1_{save_path}")
    print(f"Best model saved to {save_path} with validation loss {best_loss:.4f}")


if __name__ == "__main__":
    main()
