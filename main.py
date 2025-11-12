import os
import random
import shutil
import pandas as pd
from PIL import Image
import io
from fastapi import Form
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from models.video.prediction_sigmoid import predict_video_class as predict_vid_sigmoid
from models.video.prediction_softmax import predict_video_class as predict_vid_softmax
from models.video.prediction_cnn_lstm import predict_video_class
#from models.image.prediction_model_cnn import predict as predict_cnn
from models.image.prediction_model_sigmoid import predict as predict_sigmoid
from models.image.prediction_model_softmax import predict as predict_softmax
from models.video.prediction_emsemble import predict_ensemble_video



app = FastAPI()
app.mount("/datasets", StaticFiles(directory="datasets"), name="datasets")
# Enable CORS (to allow frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges"]
)

# **Set absolute path for datasets**
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get backend folder path
DATASET_PATH = os.path.join(BASE_DIR, "datasets")  # Absolute path to datasets
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")  # 🔥 New: Store user uploads
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create folder if not exists

# Debugging output (Check the correct path)
print("Serving datasets from:", DATASET_PATH)

# **Mount the dataset folder correctly**
if os.path.exists(DATASET_PATH):
    app.mount("/datasets", StaticFiles(directory=DATASET_PATH), name="datasets")
else:
    print(f"⚠️ Warning: Dataset folder '{DATASET_PATH}' not found!")

# Corrected file paths
image_metadata_path = os.path.join(DATASET_PATH, "images", "imagegame_metadata.csv")
video_metadata_path = os.path.join(DATASET_PATH, "videos", "videogame_metadata.csv")

# Load metadata if files exist
image_metadata = pd.read_csv(image_metadata_path) if os.path.exists(image_metadata_path) else None
video_metadata = pd.read_csv(video_metadata_path) if os.path.exists(video_metadata_path) else None

@app.get("/")
def read_root():
    return {"message": "Deepfake Detection Game API is running!"}

# Check if metadata files are loaded
@app.get("/check-metadata")
def check_metadata():
    return {
        "image_metadata_loaded": image_metadata is not None,
        "video_metadata_loaded": video_metadata is not None,
    }

# **API to get a random image**
@app.get("/game/image")
def get_random_image():
    try:
        fake_images = os.listdir(os.path.join(DATASET_PATH, "images", "fake"))
        real_images = os.listdir(os.path.join(DATASET_PATH, "images", "real"))

        if not fake_images and not real_images:
            raise HTTPException(status_code=404, detail="No images found in dataset.")

        if random.choice([True, False]) and fake_images:
            image = random.choice(fake_images)
            label = "fake"
        else:
            image = random.choice(real_images) if real_images else random.choice(fake_images)
            label = "real"

        return {"url": f"/datasets/images/{label}/{image}", "correct_label": label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# **API to get a random video**
@app.get("/game/video")
def get_random_video():
    try:
        fake_videos = os.listdir(os.path.join(DATASET_PATH, "videos", "fake"))
        real_videos = os.listdir(os.path.join(DATASET_PATH, "videos", "real"))

        if not fake_videos and not real_videos:
            raise HTTPException(status_code=404, detail="No videos found in dataset.")

        if random.choice([True, False]) and fake_videos:
            video = random.choice(fake_videos)
            label = "fake"
        else:
            video = random.choice(real_videos) if real_videos else random.choice(fake_videos)
            label = "real"

        return {
            "url": f"/serve-video/{label}/{video}",  # ✅ Correct serving path
            "correct_label": label
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/serve-video/{label}/{filename}")
def serve_video(label: str, filename: str):
    """Serve video files properly"""
    video_path = os.path.join(DATASET_PATH, "videos", label, filename)

    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video not found")

    return FileResponse(video_path, media_type="video/mp4")

@app.post("/detect-video")
async def detect_video(file: UploadFile = File(...), model: str = Form("cnn_lstm")):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 🔁 Dispatch to correct model
        if model == "softmax":
            result = predict_vid_softmax(temp_path)
        elif model == "sigmoid":
            result = predict_vid_sigmoid(temp_path)
        elif model == "ensemble":
            result = predict_ensemble_video(temp_path)
        else:
            result = predict_video_class(temp_path)

        os.remove(temp_path)

        label = result["label"]
        confidence = result["confidence"]

        return {
            "prediction": label.lower(),
            "confidences": {
                "real": float(1 - confidence) if label == "FAKE" else float(confidence),
                "fake": float(confidence) if label == "FAKE" else float(1 - confidence)
            },
            "log": result.get("log", [])  # ✅ Add this line to return logs
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...), model: str = Form(...)):
    try:
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        img = Image.open(temp_path).convert("RGB")

        if model == "inceptionresnet":
            confidences = predict_softmax(img)
        elif model == "efficientnet":
            confidences = predict_sigmoid(img)
        else:
            return {"error": "Invalid model selection."}

        os.remove(temp_path)

        return {
            "prediction": max(confidences, key=confidences.get),
            "confidences": confidences
        }

    except Exception as e:
        return {"error": str(e)}