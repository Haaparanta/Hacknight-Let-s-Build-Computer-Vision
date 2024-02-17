import io
import os
import random
from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src.train_waldo.model import trained_model

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    models["waldo"] = trained_model()
    yield


app = FastAPI(lifespan=lifespan)


# Upload file and return percentage of waldo found
@app.post("/api/waldo")
def waldo(file: Annotated[bytes, File()]):
    image = Image.open(io.BytesIO(file))
    # Edit the image to be 256x256 and grayscale
    image = image.convert("L").resize((256, 256))

    # Save the image to a file
    image.save("test.png")

    array = np.array(image)

    print(array.shape)

    model = models["waldo"]
    result = model.predict(array[np.newaxis, :, :])

    return {"percentage": float(result)}


# Static files
frontendDir = os.path.join(os.path.dirname(__file__), "../frontend")
app.mount("/", StaticFiles(directory=frontendDir, html=True), name="static")
