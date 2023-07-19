import logging
from tqdm import tqdm
import cv2
import random
import numpy as np
import os
import tensorflow as tf


def get_videos(path_list, label, num_videos):
    all_videos = []
    for path in path_list:
        videos = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.avi') and os.path.isfile(os.path.join(path, f))]
        all_videos.extend(videos)

    random.shuffle(all_videos)

    selected_videos = all_videos[:num_videos]

    video_paths = selected_videos
    video_labels = [label] * len(selected_videos)

    return video_paths, video_labels


def save_video_labels_to_file(filename, video_paths, labels):
    with open(filename, "w") as file:
        file.writelines(f"{video_path},{label}\n" for video_path, label in zip(video_paths, labels))


def process_dataset(native_videos, modified_videos, native_labels, modified_labels):
    processed_native_videos, native_videos_paths = process_videos(native_videos)
    processed_modified_videos, modified_videos_paths = process_videos(modified_videos)

    processed_videos = tf.data.Dataset.from_tensor_slices(
        np.concatenate([processed_native_videos, processed_modified_videos], axis=0).astype(np.float32) / 255.0
    )

    labels = tf.data.Dataset.from_tensor_slices(
        np.concatenate([native_labels, modified_labels], axis=0).astype(np.int32)
    )

    all_video_paths = native_videos_paths + modified_videos_paths

    return processed_videos, labels, all_video_paths


def process_videos(videos):
    processed_videos = []
    video_paths = []
    bgSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10)

    for video_path in tqdm(videos, desc='Processing videos', position=0, leave=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.warning(f"Could not open video file: {video_path}")
            continue

        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_frames = []

        for frame_count in range(num_frames):
            ret, frame = cap.read()

            if not ret or frame.size == 0:
                break

            fgMask = bgSub.apply(frame)
            mask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
            processed_frame = cv2.bitwise_and(frame, mask)

            if 50 < frame_count <= 150 and frame_count % 10 == 0:
                video_frames.append(processed_frame)

        cap.release()

        if len(video_frames) == 0:
            continue

        video_paths.append(video_path)
        video_frames = np.array(video_frames)[..., [2, 1, 0]]
        video_frames = np.maximum(video_frames, 0)
        processed_videos.append(np.stack(video_frames, axis=0))

    return processed_videos, video_paths

