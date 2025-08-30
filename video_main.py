from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import shutil
from runwayml import RunwayML

# --------------------
# FastAPI app setup
# --------------------
app = FastAPI(title="AI Generator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For demo; restrict in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------------
# RunwayML client
# --------------------
runway_client = RunwayML(api_key=os.getenv("RUNWAY_API_KEY"))

# --------------------
# Image-to-Video route
# --------------------
@app.post("/image-to-video/")
async def image_to_video(
    prompt: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Generates a video from a single image and prompt.
    Returns a downloadable video for the Image-to-Video page only.
    """
    temp_image_path = None
    temp_video_path = None

    try:
        # Save uploaded image temporarily
        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        with open(temp_image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        # Create a temporary path for the output video
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        # Generate video using RunwayML
        runway_client.video_from_image(
            image_path=temp_image_path,
            prompt=prompt,
            output_path=temp_video_path
        )

        return FileResponse(
            path=temp_video_path,
            media_type="video/mp4",
            filename="generated_video.mp4"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Clean up temporary image file
        try:
            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        except:
            pass

# --------------------
# Text-to-Image route
# --------------------
@app.post("/api/text-to-image")
async def text_to_image(prompt: dict):
    """
    Generates an image from text.
    """
    try:
        text_prompt = prompt.get("prompt")
        if not text_prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

        # Generate image using RunwayML
        runway_client.image.generate(
            prompt=text_prompt,
            output_path=temp_image_path
        )

        return FileResponse(
            path=temp_image_path,
            media_type="image/png",
            filename="generated_image.png"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")
