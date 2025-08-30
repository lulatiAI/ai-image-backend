from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import logging
from runwayml import RunwayML

# --------------------
# Logging setup
# --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger("ai-image-backend")

app = FastAPI(title="AI Generator API")

# Allow all CORS origins for testing/debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# --------------------
# Environment variables
# --------------------
RUNWAY_API_KEY = os.getenv("RUNWAY_API_KEY")
if not RUNWAY_API_KEY:
    logger.error("RUNWAY_API_KEY is missing in environment variables.")
    raise RuntimeError("RUNWAY_API_KEY is not set")

logger.info("RUNWAY_API_KEY loaded successfully.")

# Initialize RunwayML client
runway_client = RunwayML(api_key=RUNWAY_API_KEY)

# --------------------
# Routes
# --------------------
@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "AI Image Backend is running."}

@app.post("/api/text-to-image")
async def text_to_image(request: Request):
    logger.info("Incoming request to /api/text-to-image")
    try:
        # Parse JSON
        data = await request.json()
        logger.info(f"Request payload: {data}")

        prompt = data.get("prompt")
        if not prompt:
            logger.warning("No prompt provided in request.")
            return JSONResponse(
                status_code=400,
                content={"error": "Prompt is required"}
            )

        # Generate temporary image path
        temp_image_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
        logger.info(f"Generated temp path: {temp_image_path}")

        # Generate image with RunwayML
        try:
            logger.info(f"Sending prompt to RunwayML: {prompt}")
            runway_client.image_from_text(prompt=prompt, output_path=temp_image_path)
            logger.info("Image generation successful.")
        except Exception as e:
            logger.error(f"RunwayML API error: {e}")
            raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

        return FileResponse(
            path=temp_image_path,
            media_type="image/png",
            filename="generated_image.png"
        )

    except Exception as e:
        logger.exception("Unhandled server error")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
