import os
import json
import cv2


def time_to_frame(time, fps):
    return int(time * fps)


def get_video_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError(f"Unable to open video file: {video_path}")
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count


def process_json_data(data, total_frames, fps=30):
    new_data = []
    for item in data:
        if item["type"] == "drag":
            frame = time_to_frame(item["time"], fps)
            new_data.append(
                {
                    "frame": frame,
                    "type": "drag_start",
                    "position": item["start_position"],
                }
            )
            new_data.append(
                {"frame": frame, "type": "drag_end", "position": item["end_position"]}
            )
        elif item["type"] == "scroll":
            start_frame = time_to_frame(item["time"], fps)
            end_frame = time_to_frame(item["end_time"], fps)
            frames_count = end_frame - start_frame
            amount_per_frame = item["amount"] / frames_count if frames_count > 0 else 0

            for frame in range(start_frame, end_frame + 1):
                new_data.append(
                    {
                        "frame": frame,
                        "type": "scroll",
                        "position": item["position"],
                        "amount": amount_per_frame,
                    }
                )
        elif item["type"] == "keypress":
            key = item["key"]
            start_frame = time_to_frame(item["start_time"], fps)
            end_frame = time_to_frame(item["end_time"], fps)
            frames_count = end_frame - start_frame
            for i, char in enumerate(key):
                event_frame = start_frame + int(i * frames_count / (len(key) - 1))
                new_data.append({"frame": event_frame, "type": char})
        else:
            frame = time_to_frame(item["time"], fps)
            item["frame"] = frame
            del item["time"]
            if item["type"] == "click":
                item["type"] = item["button"]
                del item["button"]
            if item["type"] == "special_key":
                item["type"] = item["key"]
                del item["key"]
            new_data.append(item)

    # ffmpeg开始记录有延迟，这个尝试去掉开始录制视频前记录的东西
    start_index = 0
    for i in range(len(new_data) - 1):
        if data[i]["frame"] > data[i + 1]["frame"]:
            start_index = i + 1
            break
    new_data = new_data[start_index:]

    i = 0
    while i < len(new_data) - 1:
        if (
            new_data[i]["type"] == "Button.left"
            and new_data[i + 1]["type"] == "Button.left"
            and new_data[i]["position"] == new_data[i + 1]["position"]
        ):
            new_data[i]["type"] = "DOUBLE_CLICK"
            del new_data[i + 1]
        else:
            i += 1

    frames_data = [
        {"frame": i, "type": None, "position": None} for i in range(total_frames)
    ]

    for item in new_data:
        frames_data[item["frame"]] = item

    return frames_data


def main(logs_folder, videos_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for log_filename in os.listdir(logs_folder):
        if log_filename.endswith(".json"):
            base_name = log_filename[:-5]  # Remove '.json' extension
            video_filename = f"video_{base_name}.mp4"
            video_path = os.path.join(videos_folder, video_filename)
            log_path = os.path.join(logs_folder, log_filename)

            total_frames = get_video_frame_count(video_path)

            with open(log_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            processed_data = process_json_data(data, total_frames)

            target_file_path = os.path.join(target_folder, log_filename)
            with open(target_file_path, "w", encoding="utf-8") as file:
                json.dump(processed_data, file, indent=4)


if __name__ == "__main__":
    logs_folder = "path/to/logs"
    videos_folder = "path/to/videos"
    target_folder = "path/to/target/folder"
    main(logs_folder, videos_folder, target_folder)
