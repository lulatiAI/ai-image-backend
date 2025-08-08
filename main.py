from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import os
import requests
import io
from dotenv import load_dotenv

# Import RunwayML SDK and error
from runwayml import RunwayML, TaskFailedError

# Load environment variables
load_dotenv()

# Setup FastAPI app
app = FastAPI()

# Configure CORS (adjust your frontend URL as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.lulati.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")
if not RUNWAY_API_KEY:
    raise RuntimeError("RUNWAY_API_KEY not set in environment")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize RunwayML client
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

# --- Image generation endpoint (unchanged) ---
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

# --- Fixed video generation endpoint ---
@app.post("/generate-video")
async def generate_video(data: PromptRequest = Body(...)):
    prompt = data.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        # Create a video generation task on RunwayML
        task = runway_client.image_to_video.create(
            model="gen4_turbo",
            prompt_text=prompt,
            ratio="1024:1024",
            duration=4
        ).wait_for_task_output()

        video_url = task.get("output", {}).get("video")
        if not video_url:
            raise HTTPException(status_code=500, detail="No video URL returned from RunwayML.")

        return {"video_url": video_url}

    except TaskFailedError as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e.task_details}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Optional: Image to video endpoint (if you use it) ---
@app.post("/image-to-video")
async def image_to_video(data: ImageToVideoRequest):
    try:
        task = runway_client.image_to_video.create(
            model="gen4_turbo",
            prompt_image=data.image_url,
            prompt_text=data.prompt,
            ratio="1024:1024",
            duration=4
        ).wait_for_task_output()

        video_url = task.get("output", {}).get("video")
        if not video_url:
            raise HTTPException(status_code=500, detail="No video URL returned from RunwayML.")

        return {"video_url": video_url}

    except TaskFailedError as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {e.task_details}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
