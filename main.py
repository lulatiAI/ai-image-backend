from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env if available
load_dotenv()

# FastAPI app
app = FastAPI()

# CORS settings - restrict to your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.lulati.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# RunwayML API key from env
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")

# --- Models ---
class PromptRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"  # optional for image

# --- Routes ---
@app.get("/")
def read_root():
    return {"message": "Hello from DALL·E and RunwayML!"}


@app.post("/generate")
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


@app.post("/generate-video")
async def generate_video(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")

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

        # Get the video URL
        video_url = r_data.get("output", {}).get("video", "")

        if not video_url:
            raise HTTPException(status_code=500, detail="Video generation failed or returned empty URL")

        return {
            "reply": f"<video controls width='100%'><source src='{video_url}' type='video/mp4'></video>"
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"RunwayML API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
