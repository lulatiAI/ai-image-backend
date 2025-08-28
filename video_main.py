from fastapi import FastAPI, HTTPException, Body, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from runwayml import RunwayML
import boto3
import requests
import uuid

load_dotenv()

app = FastAPI()

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for testing; replace with your domain in prod
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

class ImageRequest(BaseModel):
    prompt: str
    ratio: str = "1360:768"
    model: str = "gen4_image"

@app.post("/generate-image")
async def generate_image(data: ImageRequest = Body(...)):
    logs = []
    prompt = data.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    blocked_words = ["nude", "sex", "violence", "gore"]
    if any(word in prompt.lower() for word in blocked_words):
        raise HTTPException(status_code=403, detail="Prompt contains inappropriate content")

    try:
        logs.append("Starting text-to-image generation...")
        task = client.text_to_image.create(
            model=data.model,
            prompt_text=prompt,
            ratio=data.ratio
        )

        logs.append("Waiting for RunwayML task to complete...")
        task_result = task.wait_for_task_output()
        logs.append("RunwayML task completed.")

        task_output = task_result.output
        if not task_output or not hasattr(task_output[0], "uri"):
            raise HTTPException(status_code=500, detail="No image returned from RunwayML.")

        output_image_url = task_output[0].uri
        logs.append(f"Image URL received: {output_image_url}")

        logs.append("Fetching image bytes...")
        img_resp = requests.get(output_image_url)
        img_resp.raise_for_status()
        image_bytes = img_resp.content
        logs.append("Image fetched successfully.")

        logs.append("Running AWS Rekognition moderation...")
        rekog_resp = rekognition.detect_moderation_labels(
            Image={"Bytes": image_bytes},
            MinConfidence=70
        )
        if rekog_resp["ModerationLabels"]:
            labels = [l["Name"] for l in rekog_resp["ModerationLabels"]]
            raise HTTPException(status_code=403, detail=f"Image flagged: {labels}")
        logs.append("Moderation passed.")

        # Save temporarily for download
        temp_filename = f"/tmp/{uuid.uuid4()}.png"
        with open(temp_filename, "wb") as f:
            f.write(image_bytes)
        logs.append("Image saved for download.")

        return {
            "image_url": output_image_url,
            "download_url": f"/download/{os.path.basename(temp_filename)}",
            "logs": logs
        }

    except Exception as e:
        logs.append(f"Error: {str(e)}")
        return {"error": str(e), "logs": logs}


@app.get("/download/{filename}")
def download_image(filename: str):
    path = f"/tmp/{filename}"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, "rb") as f:
        return StreamingResponse(
            f,
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
