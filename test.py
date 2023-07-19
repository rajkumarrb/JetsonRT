import cv2
import glob
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from keras.models import load_model
from paths import native_paths, modified_paths

test_path = [native_paths, modified_paths]

test_size = 10
test_path = test_path[0]

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def preprocess(frame):
    bgSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10)
    fgMask = bgSub.apply(frame)
    mask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
    processed_frame = cv2.bitwise_and(frame, mask)
    return processed_frame


model = load_model("model_0719_1901_epochs_50_videos_100.keras")

video_files = sum((glob.glob(path + "/*.avi") for path in test_path), [])
video_files = np.random.choice(video_files, size=test_size, replace=False)

processed_videos = []
video_paths_list = []

true_labels = [1] * len(video_files) if test_path == native_paths else [0] * len(video_files)
predicted_labels = []
class_labels = ["Modified", "Native"]

for video_file in tqdm(video_files, desc='Processing videos', position=0, leave=True):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        continue

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frames = []

    for frame_count in range(num_frames):
        ret, frame = cap.read()

        if not ret or frame.size == 0:
            break

        processed_frame = preprocess(frame)
        if 50 < frame_count <= 150 and frame_count % 10 == 0:
            video_frames.append(processed_frame)

    cap.release()

    if len(video_frames) == 0:
        continue

    video_paths_list.append(video_file)
    video_frames = np.array(video_frames)
    processed_videos.append(video_frames)

processed_videos = np.array(processed_videos).astype(np.float32) / 255.0
processed_videos_dataset = tf.data.Dataset.from_tensor_slices(processed_videos)

for video_tensor, video_path, true_label in tqdm(zip(processed_videos_dataset, video_paths_list, true_labels), desc='Predicting labels', position=0, leave=True):
    video_tensor = tf.reshape(video_tensor, (1,) + video_tensor.shape)
    predictions = model.predict(video_tensor)
    predicted_class = np.argmax(predictions)
    predicted_labels.append(predicted_class)
    tqdm.write("Predicted Class: {}".format(class_labels[predicted_class]))

accuracy = np.mean(np.equal(true_labels, predicted_labels))
print("\nAccuracy: {:.2%}\n".format(accuracy))

