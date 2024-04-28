import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np

FRAME_WIDTH = None
FRAME_HEIGHT = None

# 定义数据集
class ActionDataset(Dataset):
    def __init__(self, video_dir, json_dir, transform=None):
        self.video_dir = video_dir
        self.json_dir = json_dir
        self.transform = transform
        self.videos = os.listdir(video_dir)
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = os.path.join(self.video_dir, self.videos[idx])
        json_path = os.path.join(self.json_dir, self.videos[idx].replace('.mp4', '.json'))
        
        # 读取视频和对应的动作数据
        video, video_len = self.load_video(video_path)
        actions = self.load_actions(json_path, video_len)
        
        if self.transform:
            video = self.transform(video)
        
        return video, actions
    
    def load_video(self, path):
        # 读取视频帧并进行预处理
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        video_len = len(frames)
        return np.array(frames), video_len
    
    def load_actions(self, path, video_len):
        # 读取动作数据并进行预处理
        with open(path, 'r') as f:
            actions = json.load(f)
        
        action_labels = np.zeros((video_len, 5))  # 5个动作类别
        for action in actions:
            start_frame = int(action['start_time'] * 30)  # 假设视频帧率为30
            end_frame = int(action['end_time'] * 30)
            
            if action['type'] == 'click':
                x = int(action['position']['x'] * 640)  # 假设视频宽度为640
                y = int(action['position']['y'] * 480)  # 假设视频高度为480
                action_labels[start_frame:end_frame, 0] = 1
                action_labels[start_frame:end_frame, 1] = x / 640.0
                action_labels[start_frame:end_frame, 2] = y / 480.0
            elif action['type'] == 'scroll':
                action_labels[start_frame:end_frame, 3] = 1
            elif action['type'] == 'drag':
                start_x = int(action['start']['x'] * 640)
                start_y = int(action['start']['y'] * 480)
                end_x = int(action['end']['x'] * 640)  
                end_y = int(action['end']['y'] * 480)
                action_labels[start_frame:end_frame, 4] = 1
                action_labels[start_frame:end_frame, 1] = np.linspace(start_x, end_x, end_frame-start_frame) / 640.0
                action_labels[start_frame:end_frame, 2] = np.linspace(start_y, end_y, end_frame-start_frame) / 480.0
        
        return action_labels

# 定义3DCNN模型
class C3D(nn.Module):
    def __init__(self, num_classes):
        super(C3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        x = x.view(-1, 8192)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))  
        x = self.dropout(x)
        logits = self.fc8(x)
        return logits

# 定义训练函数
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for videos, actions in dataloader:
        videos = videos.to(device)
        actions = actions.to(device)
        
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, actions)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * videos.size(0)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

# 定义测试函数
def test(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    
    with torch.no_grad():
        for videos, actions in dataloader:
            videos = videos.to(device)
            actions = actions.to(device)
            
            outputs = model(videos)
            loss = criterion(outputs, actions)
            
            running_loss += loss.item() * videos.size(0)
            running_corrects += torch.sum(torch.argmax(outputs, dim=2) == torch.argmax(actions, dim=2))
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / (len(dataloader.dataset) * actions.size(1))
    
    return epoch_loss, epoch_acc

# 设置超参数
num_epochs = 50
batch_size = 8
learning_rate = 1e-4
num_classes = 5

# 加载数据集
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_dataset = ActionDataset(train_video_dir, train_json_dir, transform=train_transform)
val_dataset = ActionDataset(val_video_dir, val_json_dir, transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = C3D(num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
best_acc = 0.0
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    val_loss, val_acc = test(model, val_dataloader, criterion, device)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')

# 在测试集上评估最优模型
model.load_state_dict(torch.load('best_model.pth'))
test_dataset = ActionDataset(test_video_dir, test_json_dir, transform=train_transform)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
test_loss, test_acc = test(model, test_dataloader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')