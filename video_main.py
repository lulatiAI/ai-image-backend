from fastapi import FastAPI, HTTPException, Body, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import os
from dotenv import load_dotenv
from runwayml import RunwayML
import boto3
import requests

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.lulati.com"],  # your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RunwayML client
client = RunwayML()

# AWS Rekognition client
rekognition = boto3.client(
    "rekognition",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

# Request model
class ImageRequest(BaseModel):
    prompt: str
    ratio: str = "1360:768"
    model: str = "gen4_image"

@app.post("/generate-image")
async def generate_image(data: ImageRequest = Body(...), download: bool = False):
    prompt = data.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    # Basic prompt moderation
    blocked_words = ["nude", "sex", "violence", "gore"]
    if any(word in prompt.lower() for word in blocked_words):
        raise HTTPException(status_code=403, detail="Prompt contains inappropriate content")

    try:
        # Call RunwayML text-to-image
        task = client.text_to_image.create(
            model=data.model,
            prompt_text=prompt,
            ratio=data.ratio
        ).wait_for_task_output()

        # Get image URL
        output_image_url = task.get("output", [{}])[0].get("uri")
        if not output_image_url:
            raise HTTPException(status_code=500, detail="No image returned from RunwayML.")

        # Fetch image bytes
        img_resp = requests.get(output_image_url)
        img_resp.raise_for_status()
        image_bytes = img_resp.content

        # AWS Rekognition moderation
        rekog_resp = rekognition.detect_moderation_labels(
            Image={"Bytes": image_bytes},
            MinConfidence=70
        )
        if rekog_resp["ModerationLabels"]:
            labels = [l["Name"] for l in rekog_resp["ModerationLabels"]]
            raise HTTPException(status_code=403, detail=f"Image flagged by moderation: {labels}")

        # Return image to frontend
        image_stream = io.BytesIO(image_bytes)
        if download:
            # Force browser to download
            headers = {"Content-Disposition": f"attachment; filename=generated_image.png"}
            return Response(content=image_bytes, media_type="image/png", headers=headers)
        else:
            # Just display image in browser
            return StreamingResponse(image_stream, media_type="image/png")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"message": "Welcome to the RunwayML Text-to-Image API"}
