from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins = origins,
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],

)

endpoint = "http://localhost:8501/v1/models/potato_disease_model:predict"
MODEL = tf.keras.models.load_model("C:/Users/kaush/OneDrive/Documents/Code/Project DL/potato_model.keras")


class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am alive!"}

# Read and preprocess image
def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")  # convert to RGB just in case
    image = image.resize((256, 256))                  # resize to model's input size
    image = np.array(image) / 255.0                   # normalize pixel values
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())     # shape: (256, 256, 3)
    img_batch = np.expand_dims(image, axis=0)         # shape: (1, 256, 256, 3)

    predictions = MODEL.predict(img_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
