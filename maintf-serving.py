from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image, ImageOps
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Load environment variables
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost,http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TF_SERVING_URL = os.getenv("TF_SERVING_URL")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data))

    # Fix EXIF rotation (for phone images)
    img = ImageOps.exif_transpose(img)

    # Convert to RGB only (3 channels)
    img = img.convert("RGB")

    return np.array(img)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)

    json_data = {"instances": img_batch.tolist()}
    response = requests.post(TF_SERVING_URL, json=json_data)

    res_json = response.json()
    print("TF Serving Response:", res_json)

    if "predictions" not in res_json:
        return {"error": "TF Serving Error", "details": res_json}

    prediction = np.array(res_json["predictions"][0])

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return {
        "class": predicted_class,
        "confidence": confidence
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host=os.getenv("API_HOST", "0.0.0.0"), 
        port=int(os.getenv("API_PORT", 8000))
    )
