import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import action_to_vec
from ActionMapping import ACTION_MAPPING


class DataLoader_nx1(Dataset):
    def __init__(self, logs_dir, videos_dir, num_frames=60, transform=None):
        self.logs_dir = logs_dir
        self.videos_dir = videos_dir
        self.num_frames = num_frames
        self.transform = transform
        self.samples = self._find_samples()

    def _find_samples(self):
        samples = []
        for log_file in os.listdir(self.logs_dir):
            if log_file.endswith(".json"):
                base_name = log_file[4:].split(".")[0]
                video_file = f"screen_{base_name}.mp4"
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

    def _merge_actions(self, actions):
        x = -1
        y = -1
        type = "NO_ACTION"
        for action in actions:
            if action["type"] != "NO_ACTION":
                type = action["type"]
                x = action["position"]["x"]
                y = action["position"]["y"]
                break
        return x, y, type

    def __getitem__(self, idx):
        base_name, start_frame = self.samples[idx]
        video_path = os.path.join(self.videos_dir, f"screen_{base_name}.mp4")
        log_path = os.path.join(self.logs_dir, f"log_{base_name}.json")

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

        with open(log_path, "r") as f:
            actions = json.load(f)

        x, y, action_type = self._merge_actions(
            actions[start_frame : start_frame + self.num_frames]
        )

        action_vec = action_to_vec(ACTION_MAPPING.get(action_type, "NO_ACTION"))
        label = np.array([x, y] + action_vec, dtype=np.float32)

        label = torch.tensor(label, dtype=torch.float32)

        return frames, label

class DataLoader_nxn(Dataset):
    def __init__(self, logs_dir, videos_dir, num_frames=60, transform=None):
        self.logs_dir = logs_dir
        self.videos_dir = videos_dir
        self.num_frames = num_frames
        self.transform = transform
        self.samples = self._find_samples()

    def _find_samples(self):
        samples = []
        for log_file in os.listdir(self.logs_dir):
            if log_file.endswith(".json"):
                base_name = log_file[4:].split(".")[0]
                video_file = f"screen_{base_name}.mp4"
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
        video_path = os.path.join(self.videos_dir, f"screen_{base_name}.mp4")
        log_path = os.path.join(self.logs_dir, f"log_{base_name}.json")

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

        with open(log_path, "r") as f:
            actions = json.load(f)

        # Assuming actions are aligned with frames
        labels = []
        for action in actions[start_frame : start_frame + self.num_frames]:
            x, y = action["position"]["x"], action["position"]["y"]
            action_type = action["type"]
            action_vec = action_to_vec(ACTION_MAPPING.get(action_type, "NO_ACTION"))
            label = np.array([x, y] + action_vec, dtype=np.float32)
            labels.append(label)

        frames = torch.stack(frames)
        labels = torch.tensor(labels, dtype=torch.float32)

        return frames, labels