import datetime
import tensorflow as tf
from keras.models import Sequential
from keras.applications import EfficientNetB0
from keras.losses import SparseCategoricalCrossentropy
from keras.layers import Rescaling, TimeDistributed, Dropout, Dense, GlobalAveragePooling3D
from sklearn.model_selection import train_test_split
from paths import native_paths, modified_paths
from processor import get_videos, process_dataset


def main():
    videos = 100
    epochs = 50

    train_ratio = 0.7
    val_ratio = 0.3

    video_index = int(videos // 2)

    current_datetime = datetime.datetime.now().strftime("%m%d_%H%M")

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    selected_native_paths = native_paths
    selected_modified_paths = modified_paths

    native_videos, native_labels = get_videos(selected_native_paths, label=1, num_videos=video_index)
    modified_videos, modified_labels = get_videos(selected_modified_paths, label=0, num_videos=video_index)

    train_native_videos, val_native_videos, train_native_labels, val_native_labels = train_test_split(
        native_videos, native_labels, train_size=train_ratio, test_size=val_ratio, random_state=42)

    train_modified_videos, val_modified_videos, train_modified_labels, val_modified_labels = train_test_split(
        modified_videos, modified_labels, train_size=train_ratio, test_size=val_ratio, random_state=42)

    train_videos_tensor, train_labels_tensor, train_vid_paths = process_dataset(train_native_videos,
                                                                                train_modified_videos,
                                                                                train_native_labels,
                                                                                train_modified_labels)
    val_videos_tensor, val_labels_tensor, val_vid_paths = process_dataset(val_native_videos,
                                                                          val_modified_videos,
                                                                          val_native_labels,
                                                                          val_modified_labels)

    autotune = tf.data.experimental.AUTOTUNE

    train_dataset = tf.data.Dataset.zip((train_videos_tensor, train_labels_tensor))
    train_dataset.cache().shuffle(10).prefetch(buffer_size=autotune)
    train_dataset = train_dataset.batch(1)

    val_dataset = tf.data.Dataset.zip((val_videos_tensor, val_labels_tensor))
    val_dataset.cache().prefetch(buffer_size=autotune)
    val_dataset = val_dataset.batch(1)

    base_model = EfficientNetB0(include_top=False)
    base_model.trainable = False

    model = Sequential([
        Rescaling(scale=255),
        TimeDistributed(base_model),
        Dropout(0.2),
        Dense(10),
        GlobalAveragePooling3D()
    ])

    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)

    model.save("models/model_{}_epochs_{}_videos_{}.keras".format(current_datetime, epochs, videos))

    print("")
    print("Finished.")
    print("")


if __name__ == "__main__":
    main()

