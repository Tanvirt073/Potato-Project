from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

model = tf.keras.models.load_model("../saved_models/3.h5")


class_names = ["Early Blight","Late Blight","Healthy"]

app = FastAPI()


@app.get("/ping")
async def ping():
    return "hello, I am alive "


def read_file_as_images(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image



@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    image = read_file_as_images(await file.read())
    image_batch = np.expand_dims(image, 0)
    predictions = model.predict(image_batch)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return {
        "class" : predicted_class,
        "confidence" : float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)