import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize

ACTION_MAPPING = {
    "a": "KEY_A",
    "b": "KEY_B",
    "c": "KEY_C",
    "d": "KEY_D",
    "e": "KEY_E",
    "f": "KEY_F",
    "g": "KEY_G",
    "h": "KEY_H",
    "i": "KEY_I",
    "j": "KEY_J",
    "k": "KEY_K",
    "l": "KEY_L",
    "m": "KEY_M",
    "n": "KEY_N",
    "o": "KEY_O",
    "p": "KEY_P",
    "q": "KEY_Q",
    "r": "KEY_R",
    "s": "KEY_S",
    "t": "KEY_T",
    "u": "KEY_U",
    "v": "KEY_V",
    "w": "KEY_W",
    "x": "KEY_X",
    "y": "KEY_Y",
    "z": "KEY_Z",
    "1": "KEY_1",
    "2": "KEY_2",
    "3": "KEY_3",
    "4": "KEY_4",
    "5": "KEY_5",
    "6": "KEY_6",
    "7": "KEY_7",
    "8": "KEY_8",
    "9": "KEY_9",
    "0": "KEY_0",
    "!": "KEY_1",
    "@": "KEY_2",
    "#": "KEY_3",
    "$": "KEY_4",
    "¥": "KEY_4",
    "%": "KEY_5",
    "^": "KEY_6",
    "&": "KEY_7",
    "*": "KEY_8",
    "(": "KEY_9",
    ")": "KEY_0",
    "（": "KEY_9",
    "）": "KEY_0",
    "-": "KEY_MINUS",
    "——": "KEY_MINUS",
    "_": "KEY_MINUS",
    "=": "KEY_EQUAL",
    "+": "KEY_EQUAL",
    '"': "KEY_QUOTE",
    "“": "KEY_QUOTE",
    "”": "KEY_QUOTE",
    "'": "KEY_QUOTE",
    "‘": "KEY_QUOTE",
    "’": "KEY_QUOTE",
    ":": "KEY_SEMICOLON",
    "：": "KEY_SEMICOLON",
    ";": "KEY_SEMICOLON",
    "；": "KEY_SEMICOLON",
    "[": "KEY_LEFTBRACE",
    "{": "KEY_LEFTBRACE",
    "]": "KEY_RIGHTBRACE",
    "}": "KEY_RIGHTBRACE",
    "【": "KEY_LEFTBRACE",
    "「": "KEY_LEFTBRACE",
    "】": "KEY_RIGHTBRACE",
    "」": "KEY_RIGHTBRACE",
    ",": "KEY_COMMA",
    "，": "KEY_COMMA",
    "<": "KEY_COMMA",
    "《": "KEY_COMMA",
    ".": "KEY_DOT",
    "。": "KEY_DOT",
    ">": "KEY_DOT",
    "》": "KEY_DOT",
    "/": "KEY_SLASH",
    "、": "KEY_SLASH",
    "?": "KEY_SLASH",
    "？": "KEY_SLASH",
    "\\": "KEY_BACKSLASH",
    "|": "KEY_BACKSLASH",
    "、": "KEY_GRAVE",
    "`": "KEY_GRAVE",
    "~": "KEY_GRAVE",
    "·": "KEY_GRAVE",
    "Key.space": "KEY_SPACE",
    "Key.shift": "KEY_SHIFT",
    "Key.ctrl": "KEY_CTRL",
    "Key.alt": "KEY_ALT",
    "Key.backspace": "KEY_BACKSPACE",
    "Key.enter": "KEY_ENTER",
    "Key.tab": "KEY_TAB",
    "Key.up": "KEY_UP",
    "Key.down": "KEY_DOWN",
    "Key.left": "KEY_LEFT",
    "Key.right": "KEY_RIGHT",
    "Key.esc": "KEY_ESC",
    "Key.cmd": "KEY_CMD",
    "Key.delete": "KEY_BACKSPACE",
    "Key.caps_lock": "KEY_CAPSLOCK",
    "Button.left": "MOUSE_LEFT_SINGLE",
    "DOUBLE_CLICK": "MOUSE_LEFT_DOUBLE",
    "Button.right": "MOUSE_RIGHT",
    "Button.middle": "MOUSE_MIDDLE",
    "drag_start": "DRAG_START",
    "drag_end": "DRAG_END",
    "scroll": "SCROLL",
}

print(len(ACTION_MAPPING.values()))


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
            if log_file.endswith(".json"):
                base_name = log_file.split("_")[1].split(".")[0]
                video_file = f"video_{base_name}.mp4"
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
        video_path = os.path.join(self.videos_dir, f"video_{base_name}.avi")
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
            x, y = action["position_x"], action["position_y"]
            action_type = action["action_type"]
            action_vec = self.action_to_vec(action_type)
            label = np.array([x, y] + action_vec, dtype=np.float32)
            labels.append(label)

        frames = torch.stack(frames)
        labels = torch.tensor(labels, dtype=torch.float32)

        return frames, labels

    def action_to_vec(self, action_type):
        action_types = ACTION_MAPPING.values()
        vec = [1 if action_type == act else 0 for act in action_types]
        return vec


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
