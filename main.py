from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import os
import requests
import io
from dotenv import load_dotenv

# Import RunwayML SDK stuff
from runwayml import RunwayML, TaskFailedError

# Load environment variables from .env
load_dotenv()

# Setup FastAPI
app = FastAPI()

# CORS settings — allow your frontend only (adjust if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.lulati.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API keys loaded from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")
if not RUNWAY_API_KEY:
    raise RuntimeError("RUNWAY_API_KEY not set in environment")

# OpenAI client initialized with the key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize RunwayML SDK client with your API key
runway_client = RunwayML(api_key=RUNWAY_API_KEY)

# --- Models ---
class PromptRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"

class ImageToVideoRequest(BaseModel):
    image_url: str
    prompt: str = ""

# --- Root endpoint ---
@app.get("/")
def read_root():
    return {"message": "Welcome to AI Media API! Running on Render."}

# --- Generate image using OpenAI DALL·E 3 (unchanged) ---
@app.post("/generate-image")
async def generate_image(data: PromptRequest):
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=data.prompt,
            n=1,
            size=data.size,
            response_format="url"
        )
        image_url = response.data[0].url
        return {"image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Generate video from text prompt using RunwayML Gen-4 Turbo via SDK ---
@app.post("/generate-video")
async def generate_video(data: PromptRequest = Body(...)):
    prompt = data.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        task = runway_client.text_to_video.create(
            model="gen4_turbo",
            prompt_text=prompt,
            duration=5,
            ratio="1280:720"
        ).wait_for_task_output()

        video_url = task.get("output", {}).get("video")
        if not video_url:
            raise HTTPException(status_code=500, detail="No video URL returned from RunwayML.")

        # Fetch video bytes and stream back
        video_response = requests.get(video_url)
        video_response.raise_for_status()
        return StreamingResponse(io.BytesIO(video_response.content), media_type="video/mp4")

    except TaskFailedError as e:
        raise HTTPException(status_code=500, detail=f"RunwayML video generation failed: {e.task_details}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Generate video from image + prompt using RunwayML Gen-4 Turbo via SDK ---
@app.post("/image-to-video")
async def image_to_video(data: ImageToVideoRequest):
    try:
        task = runway_client.image_to_video.create(
            model="gen4_turbo",
            prompt_image=data.image_url,
            prompt_text=data.prompt,
            duration=5,
            ratio="1280:720"
        ).wait_for_task_output()

        video_url = task.get("output", {}).get("video")
        if not video_url:
            raise HTTPException(status_code=500, detail="No video URL returned from RunwayML.")

        video_response = requests.get(video_url)
        video_response.raise_for_status()
        return StreamingResponse(io.BytesIO(video_response.content), media_type="video/mp4")

    except TaskFailedError as e:
        raise HTTPException(status_code=500, detail=f"RunwayML video generation failed: {e.task_details}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
