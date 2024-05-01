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
import torchvision
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_fpn,
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
)


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

        # Format for object detection
        target = {}
        target["boxes"] = torch.tensor(
            [[mouse_x, mouse_y, mouse_x + 1, mouse_y + 1]], dtype=torch.float32
        )  # Minimal box size
        target["labels"] = torch.tensor(
            [1], dtype=torch.int64
        )  # Assuming '1' is the label for mouse

        return frame, target

    def __len__(self):
        return len(self.data)


class MousePositionPredictor(nn.Module):
    def __init__(self, num_classes=2):
        super(MousePositionPredictor, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            pretrained=True
        )

        # Replace the last layer with a new linear layer for regression
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
        )

    def forward(self, x):
        return self.model(x)


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 5, 6, 7"

    batch_size = 64
    num_epochs = 5
    learning_rate = 0.0001
    train_val_split_ratio = 0.95  # 训练和验证集的比例
    save_path = "model.pth"

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights_backbone=FasterRCNN_MobileNet_V3_Large_FPN_Weights, num_classes=2
    )
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
            transforms.Resize((720, 720)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
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
