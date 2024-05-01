import os
import json
import cv2

from .src.ActionMapping import SPECIAL_KEYS

def time_to_frame(time, fps):
    return int(time * fps)


def get_video_frame_count(video_path):
    try:
        video = cv2.VideoCapture(video_path)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        video.release()
        return frame_count
    except:
        return 0


def process_json_data(data, total_frames, fps=30):
    new_data = []
    for item in data:
        if "time" in item and not item["time"]:
            continue
        if item["type"] == "drag":
            new_data.append(
                {
                    "frame": time_to_frame(item["start_time"], fps),
                    "type": "drag_start",
                    "position": item["start"],
                }
            )
            new_data.append(
                {
                    "frame": time_to_frame(item["end_time"], fps),
                    "type": "drag_end",
                    "position": item["end"],
                }
            )
        elif item["type"] == "scroll":
            start_frame = time_to_frame(item["start_time"], fps)
            end_frame = time_to_frame(item["end_time"], fps)
            frames_count = end_frame - start_frame
            amount_per_frame = item["amount"] / frames_count if frames_count > 0 else 0

            for frame in range(start_frame, end_frame + 1):
                new_data.append(
                    {
                        "frame": frame,
                        "type": "scroll_up" if amount_per_frame > 0 else "scroll_down",
                        "position": item["position"],
                        "amount": amount_per_frame,
                    }
                )
        elif item["type"] == "keypress":
            key = item["keys"]
            if len(key) == 1:
                new_data.append(
                    {"frame": time_to_frame(item["start_time"], fps), "type": key, "position": {"x": -1, "y": -1}}
                )
            else:
                start_frame = time_to_frame(item["start_time"], fps)
                end_frame = time_to_frame(item["end_time"], fps)
                frames_count = end_frame - start_frame
                for i, char in enumerate(key):
                    event_frame = start_frame + int(i * frames_count / (len(key) - 1))
                    new_data.append({"frame": event_frame, "type": char, "position": {"x": -1, "y": -1}})
        else:
            frame = time_to_frame(item["time"], fps)
            item["frame"] = frame
            del item["time"]
            if item["type"] == "click":
                item["type"] = item["button"]
                del item["button"]
            if item["type"] == "special_key":
                if item["key"] in SPECIAL_KEYS:
                    item["type"] = item["key"]
                else:
                    item["type"] = "NO_ACTION"
                item["position"] = {"x": -1, "y": -1}
                del item["key"]
            new_data.append(item)

    # ffmpeg开始记录有延迟，这个尝试去掉开始录制视频前记录的东西
    start_index = 0
    for i in range(len(new_data) - 1):
        if new_data[i]["frame"] > new_data[i + 1]["frame"]:
            start_index = i + 1
            break
    new_data = new_data[start_index:]

    i = 0
    while i < len(new_data) - 1:
        if (
            new_data[i]["type"] == "Button.left"
            and new_data[i + 1]["type"] == "Button.left"
            and new_data[i]["position"] == new_data[i + 1]["position"]
            and new_data[i + 1]["frame"] - new_data[i]["frame"] < fps
        ):
            new_data[i]["type"] = "DOUBLE_CLICK"
            del new_data[i + 1]
        else:
            i += 1

    frames_data = [
        {"frame": i, "type": "NO_ACTION", "position": {"x": -1, "y": -1}}
        for i in range(total_frames)
    ]

    for item in new_data:
        try:
            frames_data[item["frame"]] = item
        except:
            break

    return frames_data


def main(logs_folder, videos_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for log_filename in os.listdir(logs_folder):
        if log_filename.endswith(".json"):
            print(log_filename)
            base_name = log_filename[:-5]  # Remove '.json' extension
            video_filename = f"screen_{base_name[4:]}.mp4"
            video_path = os.path.join(videos_folder, video_filename)
            log_path = os.path.join(logs_folder, log_filename)

            total_frames = get_video_frame_count(video_path)

            if total_frames == 0:
                continue

            with open(log_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            processed_data = process_json_data(data, total_frames)

            target_file_path = os.path.join(target_folder, log_filename)
            with open(target_file_path, "w", encoding="utf-8") as file:
                json.dump(processed_data, file, indent=4)


if __name__ == "__main__":
    logs_folder = "/Users/bernoulli_hermes/projects/cad/detect/logs"
    videos_folder = "/Users/bernoulli_hermes/projects/cad/detect/videos"
    target_folder = "/Users/bernoulli_hermes/projects/cad/detect/logs_refined"
    main(logs_folder, videos_folder, target_folder)
