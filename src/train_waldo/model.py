import os
import numpy as np
from typing import TYPE_CHECKING

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .waldo_loader import val_ds, train_ds


if TYPE_CHECKING:
    from tensorflow.python.keras import layers
    from tensorflow.python import keras
else:
    from tensorflow.keras import layers
    from tensorflow import keras


def make_model():

    model = keras.Sequential(
        [
            keras.Input(shape=(256, 256, 1)),
            layers.Conv2D(
                10, (3, 3), strides=(2, 2), padding="valid", activation="relu"
            ),
            layers.MaxPool2D(2),
            layers.Conv2D(
                10, (3, 3), strides=(2, 2), padding="valid", activation="relu"
            ),
            layers.MaxPool2D(2),
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.summary()
    return model


def trained_model():
    train_loader = train_ds
    validation_loader = val_ds
    model = make_model()
    model.compile(optimizer="Adam", loss="binary_crossentropy")
    model.fit(
        train_loader,
        validation_data=validation_loader,
        epochs=20,
    )
    return model


if __name__ == "__main__":
    model = trained_model()
