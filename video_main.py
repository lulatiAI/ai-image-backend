# ai_generator_api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import traceback
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
    temp_image_path = None
    try:
        # Get JSON payload
        data = await request.json()
        text_prompt = data.get("prompt")
        if not text_prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        # Create a temporary file for output
        temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        temp_image_path = temp_image_file.name
        temp_image_file.close()  # Close so RunwayML can write to it

        # Generate image using RunwayML
        try:
            runway_client.image.generate(prompt=text_prompt, output_path=temp_image_path)
        except Exception as e:
            print("RunwayML generation error:")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

        # Return image file
        return FileResponse(
            path=temp_image_path,
            media_type="image/png",
            filename="generated_image.png"
        )

    except HTTPException:
        raise
    except Exception as e:
        print("Internal server error:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    finally:
        # Optional: schedule temp file cleanup if needed
        pass
