import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

class VideoActionDataset(Dataset):
    def __init__(self, logs_dir, videos_dir, num_frames=60, transform=None):
        self.logs_dir = logs_dir
        self.videos_dir = videos_dir
        self.num_frames = num_frames
        self.transform = transform
        self.samples = self._find_samples()

    def _find_samples(self):
        samples = []
        for log_file in os.listdir(self.logs_dir):
            if log_file.endswith('.json'):
                base_name = log_file.split('_')[1].split('.')[0]
                video_file = f'video_{base_name}.avi'
                video_path = os.path.join(self.videos_dir, video_file)
                if os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    # Generate samples for each set of 60 frames, with a step of 1 frame
                    for i in range(total_frames - self.num_frames + 1):
                        samples.append((base_name, i))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_name, start_frame = self.samples[idx]
        video_path = os.path.join(self.videos_dir, f'video_{base_name}.avi')
        log_path = os.path.join(self.logs_dir, f'log_{base_name}.json')
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        
        with open(log_path, 'r') as f:
            actions = json.load(f)
        
        # Assuming actions are aligned with frames
        labels = []
        for action in actions[start_frame:start_frame+self.num_frames]:
            x, y = action['position_x'], action['position_y']
            action_type = action['action_type']
            action_vec = self.action_to_vec(action_type)
            label = np.array([x, y] + action_vec, dtype=np.float32)
            labels.append(label)
        
        frames = torch.stack(frames)
        labels = torch.tensor(labels, dtype=torch.float32)

        return frames, labels

    def action_to_vec(self, action_type):
        action_types = ['no action', 'single click', 'double click', 'ctrl', 'backspace', 'a', 'b', 'c']
        vec = [1 if action_type == act else 0 for act in action_types]
        return vec

transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = VideoActionDataset(logs_dir='path/to/logs', videos_dir='path/to/videos', num_frames=60, transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)