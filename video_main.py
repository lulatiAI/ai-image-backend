from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Adjust CORS to your frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.lulati.com"],  # change to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
if not RUNWAY_API_KEY:
    raise RuntimeError("RUNWAY_API_KEY not set in environment")

class VideoRequest(BaseModel):
    prompt: str

@app.post("/generate-video")
async def generate_video(data: VideoRequest = Body(...)):
    prompt = data.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        payload = {
            "model": "gen2",
            "input": {
                "prompt": prompt,
                "duration": 4  # seconds, adjust if needed
            }
        }
        headers = {
            "Authorization": f"Bearer {RUNWAY_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post("https://api.runwayml.com/v1/generate", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        video_url = data.get("output", {}).get("video", "")
        if not video_url:
            raise HTTPException(status_code=500, detail="No video URL returned from RunwayML.")

        # Stream the video to the client
        video_resp = requests.get(video_url, stream=True)
        video_resp.raise_for_status()

        def iterfile():
            for chunk in video_resp.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk

        return StreamingResponse(iterfile(), media_type="video/mp4")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"RunwayML API error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the RunwayML Text-to-Video API"}
