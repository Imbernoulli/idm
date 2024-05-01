import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset


class MouseMoveDataset(Dataset):
    def __init__(
        self, video_folder, log_folder, transform=None, device=torch.device("cpu")
    ):
        self.video_folder = video_folder
        self.log_folder = log_folder
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
            video_name = log_file.replace("log", "screen")[:-5]
            video_path = self.video_files.get(video_name)
            if video_path is None:
                continue  # Skip if no corresponding video found

            with open(os.path.join(self.log_folder, log_file), "r") as f:
                log_data = json.load(f)
            self.mouse_move_events.extend(
                [
                    (video_path, event)
                    for event in log_data
                    if event["type"] == "mouse_move"
                ]
            )

    def __getitem__(self, idx):
        video_path, event = self.mouse_move_events[idx]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_index = int(event["time"] * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        height, width, _ = frame.shape
        print(frame.shape)

        # Convert normalized positions to pixel positions
        mouse_position = torch.tensor(
            [event["position"]["x"] * width, event["position"]["y"] * height],
            dtype=torch.float32,
        ).to(
            self.device
        )  # Move mouse_position to specified device

        print(mouse_position)

        return frame, mouse_position

    def __len__(self):
        return len(self.mouse_move_events)


def main():
    # Parameters
    video_folder = "cursor_data"
    log_folder = "cursor_data"
    output_folder = "output_frames"
    os.makedirs(output_folder, exist_ok=True)

    # Define dataset
    dataset = MouseMoveDataset(video_folder, log_folder, device=torch.device("cuda"))

    # Process each item in the dataset
    for index in range(len(dataset)):
        frame, mouse_pos = dataset[index]
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving
        x, y = int(mouse_pos[0]), int(mouse_pos[1])
        print(x, y)
        cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)  # Draw circle at mouse position

        # Save frame
        output_path = os.path.join(output_folder, f"frame_{index}.jpg")
        cv2.imwrite(output_path, frame)


if __name__ == "__main__":
    main()
