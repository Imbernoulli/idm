import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize
from .DataHandler import VideoActionDataset
from .Model import Simple3DCNN

def main():
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化数据集和数据加载器
    transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = VideoActionDataset(logs_dir='path/to/logs', videos_dir='path/to/videos', num_frames=60, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 初始化模型
    model = Simple3DCNN(num_classes=10)
    model.to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    num_epochs = 2  # 对于测试，我们仅运行少量的epoch
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:  # 每10个batch打印一次
                print(f'Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss / 10}')
                running_loss = 0.0

    print('Finished Training')

if __name__ == "__main__":
    main()