import json
import cv2
import random

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return fps, width, height, total_frames


def convert_txt(video_folder, txt_file1, txt_file2):

    json_data = []
    event_class = []

    with open(txt_file2, 'r') as f:
        for line in f:
            event_class.append(line.strip())
        f.close()

    with open(txt_file1, 'r') as f:
        for line in f:
            element = line.strip().split(' ')
            video_name = element[0]
            frame_num = list(map(int, element[1:]))

            video_path = f'{video_folder}/{video_name}.mp4'
            video_info = get_video_info(video_path)
            if video_info is None:
                continue

            fps, width, height, total_frames = video_info

            events = []
            for i in range(len(frame_num)):
                events.append({
                    "frame": frame_num[i],
                    "label": event_class[i]
                })

            video_data = {
                "video": video_name,
                "num_frames": total_frames,
                "num_events": len(events),
                "events": events,
                "fps": fps,
                "width": width,
                "height": height
            }
            # print(video_data)
            json_data.append(video_data)

    return json_data

dataset = 'ff'
txt_file1 = rf'data\{dataset}\label.txt'
txt_file2 = rf'data\{dataset}\class.txt'
video_folder = rf'data\{dataset}\{dataset}_videos'

result = convert_txt(video_folder, txt_file1, txt_file2)

with open(rf'data\{dataset}\data.json', 'w') as json_file:
    json.dump(result, json_file, indent=4)

with open(rf'data\{dataset}\data.json', 'r') as file:
    data = json.load(file)

random.shuffle(data)

total_videos = len(data)
train_size = int(0.8 * total_videos)
val_size = int(0.1 * total_videos)
test_size = total_videos - train_size - val_size

train_data = data[:train_size]
val_data = data[train_size:train_size+val_size]
test_data = data[train_size+val_size:]

with open(rf'data\{dataset}\train.json', 'w') as file:
    json.dump(train_data, file, indent=4)

with open(rf'data\{dataset}\val.json', 'w') as file:
    json.dump(val_data, file, indent=4)

with open(rf'data\{dataset}\test.json', 'w') as file:
    json.dump(test_data, file, indent=4)

