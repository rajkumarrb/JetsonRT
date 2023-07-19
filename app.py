import os
import cv2
import h5py
import gradio as gr
import numpy as np
import tensorflow as tf


def preprocess_video_frame(frame):
    bgSub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=10)
    fgMask = bgSub.apply(frame)
    mask = cv2.cvtColor(fgMask, cv2.COLOR_GRAY2BGR)
    processed_frame = cv2.bitwise_and(frame, mask)
    return processed_frame


def predict_class(video):
    # Load the video
    cap = cv2.VideoCapture(video)

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_frames = []

    # Preprocess the video data
    for frame_count in range(num_frames):
        ret, frame = cap.read()

        processed_frame = preprocess_video_frame(frame)
        video_frames.append(processed_frame)

    cap.release()

    video_frames = np.array(video_frames).astype("float32") / 255.0
    processed_video_dataset = tf.data.Dataset.from_tensor_slices(video_frames)

    # Make a prediction
    video_data = tf.reshape(processed_video_dataset, (1,) + processed_video_dataset.shape)
    prediction = model.predict(video_data)
    prediction = np.argmax(prediction)

    # Return the prediction
    return prediction[0]


# Load the trained TensorFlow model
model = tf.keras.models.load_model("model.keras")

# Create a SavedModelBuilder
builder = tf.saved_model.builder.SavedModelBuilder("./saved_model")

# Add the model to the SavedModelBuilder
builder.add_meta_graph_and_variables(
    sess=tf.keras.backend.get_session(), tags=[tf.saved_model.SERVING], signature_def_map=model._saved_model_signature_def_map
)

# Save the SavedModel
builder.save()

# Load the SavedModel
model = h5py.File("./saved_model/model.h5", "r")

# Create a list of videos to select from
videos = []
for video in os.listdir("./videos"):
    if video.endswith(".avi"):
        videos.append(video)

# Create a Gradio interface with a dropdown selection for the video and a text output for the prediction
app = gr.Interface(fn=predict_class, inputs="select", outputs="text", title="Video Classifier")

# Launch the Gradio app
app.launch()
