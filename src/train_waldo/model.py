import os
import numpy as np
from typing import TYPE_CHECKING

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if TYPE_CHECKING:
    from tensorflow.python.keras import layers
    from tensorflow.python import keras
else:
    from tensorflow.keras import layers
    from tensorflow import keras


def make_model():

    model = keras.Sequential(
        [
            keras.Input(shape=(64, 64, 1)),
            layers.Conv2D(
                10, (3, 3), strides=(2, 2), padding="valid", activation="relu"
            ),
            layers.MaxPool2D(2),
            layers.Conv2D(
                10, (3, 3), strides=(2, 2), padding="valid", activation="relu"
            ),
            layers.MaxPool2D(2),
            layers.Flatten(),
            layers.Dense(2, activation="sigmoid"),
        ]
    )
    model.summary()
    return model


def trained_model(train_loader, validation_loader):
    model = make_model()
    model.compile(optimizer="Adam", loss="mse")
    model.fit(
        train_loader,
        validation_data=validation_loader,
        epochs=20,
    )
