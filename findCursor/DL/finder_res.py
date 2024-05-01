import json
import os
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "4,7"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, random_split
from torchvision import transforms
import numpy as np
import torch.nn.functional as F


# 定义自定义数据集
class MouseMoveDataset(Dataset):
    def __init__(
        self, video_folder, log_folder, transform=None, device=torch.device("cpu")
    ):
        self.video_folder = video_folder
        self.log_folder = log_folder
        self.output_folder = "output"  # 新增保存标注图像的文件夹
        self.transform = transform
        self.device = device

        self.video_files = {
            f[:-4]: os.path.join(video_folder, f)
            for f in os.listdir(video_folder)
            if f.endswith(".mp4")
        }
        self.log_files = [f for f in os.listdir(log_folder) if f.endswith(".json")]

        self.mouse_move_events = []
        for log_file in self.log_files:
            with open(os.path.join(self.log_folder, log_file), "r") as f:
                log_data = json.load(f)
            self.mouse_move_events.extend(
                [
                    (log_file[:-5], event)
                    for event in log_data
                    if event["type"] == "mouse_move"
                ]
            )

    def __getitem__(self, idx):
        video_name, event = self.mouse_move_events[idx]
        video_path = self.video_files.get(video_name.replace("log", "screen"))
        if video_path is None:
            raise FileNotFoundError(f"Video file not found for {video_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        frame_time = min(event["time"], video_duration)

        cap.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000)
        ret, frame = cap.read()
        if ret:
            frame_height, frame_width = frame.shape[:2]
            x, y = int(event["position"]["x"] * frame_width), int(
                event["position"]["y"] * frame_height
            )
            cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)  # 在鼠标位置处画一个红色圆点
            print(frame.shape)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = np.zeros((720, 720, 3), dtype=np.uint8)

        print(frame.shape)
        # 保存带有标注的图像
        output_image_path = os.path.join(self.output_folder, f"{video_name}_{idx}.png")
        cv2.imwrite(output_image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if self.transform:
            frame = self.transform(frame)

        frame = frame.to(self.device)  # Move frame to specified device

        mouse_position = torch.tensor([x, y], dtype=torch.float32).to(
            self.device
        )  # Move mouse_position to specified device

        cap.release()

        return frame, mouse_position

    def __len__(self):
        return len(self.mouse_move_events)


class ResidualBlock(nn.Module):
    """实现具有残差连接的卷积块"""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class MousePositionPredictor(nn.Module):
    def __init__(self):
        super(MousePositionPredictor, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.res1 = ResidualBlock(16, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res2 = ResidualBlock(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# 训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total_batches = len(dataloader)

    for i, (images, mouse_positions) in enumerate(dataloader):
        images = images.to(device)
        mouse_positions = mouse_positions.to(device)

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
            images = images.to(device)
            mouse_positions = mouse_positions.to(device)

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
    # 设置参数
    video_folder = "cursor_data"
    log_folder = "cursor_data"
    batch_size = 128
    num_epochs = 2
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
            transforms.Resize((720, 720)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = MouseMoveDataset(
        video_folder, log_folder, transform=transform, device=device
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
    torch.save(best_model_state_dict, save_path)
    print(f"Best model saved to {save_path} with validation loss {best_loss:.4f}")


if __name__ == "__main__":
    main()
