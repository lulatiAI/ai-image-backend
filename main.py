from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import os
from runwayml import RunwayML, TaskFailedError

app = FastAPI()

openai_api_key = os.getenv("OPENAI_API_KEY")
runway_api_key = os.getenv("RUNWAY_API_KEY")

client = RunwayML(api_key=runway_api_key)

class ImageRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"

class ImageToVideoRequest(BaseModel):
    image_url: str
    prompt: str = ""
    ratio: str = "16:9"  # Default ratio added here

@app.post("/generate")
def generate_image(data: ImageRequest):
    # Image generation (unchanged)
    headers = {
        "Authorization": f"Bearer {openai_api_key}"
    }
    json_data = {
        "prompt": data.prompt,
        "n": 1,
        "size": data.size
    }
    response = requests.post("https://api.openai.com/v1/images/generations", headers=headers, json=json_data)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Image generation failed.")
    image_url = response.json()['data'][0]['url']
    return {"image_url": image_url}

@app.post("/image-to-video")
def image_to_video(data: ImageToVideoRequest):
    try:
        payload = {
            "model": "gen2",
            "input": {
                "image": data.image_url,
                "prompt": data.prompt,
                "duration": 4,
                "ratio": data.ratio  # Fixed missing ratio param here
            }
        }
        task = client.image_to_video.create(**payload).wait_for_task_output()
        return {"video_url": task["output"]["url"]}
    except TaskFailedError as e:
        raise HTTPException(status_code=500, detail="Video generation failed.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
