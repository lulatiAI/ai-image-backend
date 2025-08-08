from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup FastAPI
app = FastAPI()

# CORS settings — only allow your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.lulati.com"],  # change if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")

# OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# --- Models ---
class PromptRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"  # Optional, for image size

class ImageToVideoRequest(BaseModel):
    image_url: str
    prompt: str = ""  # Optional — you can send empty

# --- Root ---
@app.get("/")
def read_root():
    return {"message": "Welcome to AI Media API! Running on Render."}

# --- Image Generation (DALL·E 3) ---
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

# --- Text to Video (RunwayML Gen-2) ---
@app.post("/generate-video")
async def generate_video(data: PromptRequest = Body(...)):
    prompt = data.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        payload = {
            "model": "gen2",
            "input": {
                "prompt": prompt,
                "duration": 4  # seconds
            }
        }

        headers = {
            "Authorization": f"Bearer {RUNWAY_API_KEY}",
            "Content-Type": "application/json"
        }

        r = requests.post("https://api.runwayml.com/v1/generate", json=payload, headers=headers)
        r.raise_for_status()
        r_data = r.json()

        video_url = r_data.get("output", {}).get("video", "")
        if not video_url:
            raise HTTPException(status_code=500, detail="No video URL returned from RunwayML.")

        return {
            "video_url": video_url,
            "reply": f"<video controls width='100%'><source src='{video_url}' type='video/mp4'></video>"
        }

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"RunwayML API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Image to Video (RunwayML Gen-2) ---
@app.post("/image-to-video")
async def image_to_video(data: ImageToVideoRequest):
    try:
        payload = {
            "model": "gen2",
            "input": {
                "image": data.image_url,
                "prompt": data.prompt,
                "duration": 4
            }
        }

        headers = {
            "Authorization": f"Bearer {RUNWAY_API_KEY}",
            "Content-Type": "application/json"
        }

        r = requests.post("https://api.runwayml.com/v1/generate", json=payload, headers=headers)
        r.raise_for_status()
        r_data = r.json()

        video_url = r_data.get("output", {}).get("video", "")
        if not video_url:
            raise HTTPException(status_code=500, detail="No video URL returned.")

        return {
            "video_url": video_url,
            "reply": f"<video controls width='100%'><source src='{video_url}' type='video/mp4'></video>"
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"RunwayML API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
