# ai_generator_api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
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
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
if not RUNWAY_API_KEY:
    raise RuntimeError("RUNWAY_API_KEY is not set in environment variables")

runway_client = RunwayML(api_key=RUNWAY_API_KEY)

# --------------------
# Text-to-Image route
# --------------------
@app.post("/api/text-to-image")
async def text_to_image(request: Request):
    """
    Generates an image from a text prompt.
    Returns a downloadable PNG image.
    """
    try:
        # Parse JSON body
        data = await request.json()
        text_prompt = data.get("prompt")
        if not text_prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # Temporary file to save generated image
        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

        # --------------------
        # Generate image with RunwayML
        # --------------------
        try:
            runway_client.image_from_text(prompt=text_prompt, output_path=temp_image_path)
        except Exception as e:
            print(f"RunwayML error: {e}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

        # Return generated image
        return FileResponse(
            path=temp_image_path,
            media_type="image/png",
            filename="generated_image.png"
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
