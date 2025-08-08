from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
from runwayml import RunwayML, TaskFailedError

app = FastAPI()

# CORS setup (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RunwayML client
client = RunwayML()

# Set your OpenAI API key here or load from env
openai.api_key = "your-openai-api-key"

# === IMAGE GENERATION - KEEP THIS EXACTLY AS YOU HAVE IT ===
@app.post("/generate")
async def generate_image(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        return JSONResponse(status_code=400, content={"detail": "Prompt is required."})

    try:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        image_url = response['data'][0]['url']
        return {"image_url": image_url}

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": "Image generation failed.", "error": str(e)})


# === TEXT-TO-VIDEO GENERATION ===
@app.post("/generate-video")
async def generate_video(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    if not prompt:
        return JSONResponse(status_code=400, content={"detail": "Prompt is required."})

    try:
        task = client.text_to_video.create(
            model="gen4_turbo",
            promptText=prompt,
            ratio="16:9",
            duration=5,
        ).wait_for_task_output()

        return {"video_url": task.output_url}

    except TaskFailedError as e:
        return JSONResponse(status_code=500, content={"detail": "Video generation failed.", "error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": "Unexpected error during video generation.", "error": str(e)})


# === IMAGE-TO-VIDEO GENERATION ===
@app.post("/image-to-video")
async def image_to_video(request: Request):
    data = await request.json()
    image_url = data.get("image_url")
    prompt_text = data.get("prompt")
    ratio = data.get("ratio", "16:9")

    if not image_url or not prompt_text:
        return JSONResponse(status_code=400, content={"detail": "Both image_url and prompt are required."})

    try:
        task = client.image_to_video.create(
            model="gen4_turbo",
            promptImage=image_url,
            promptText=prompt_text,
            ratio=ratio,
            duration=5,
        ).wait_for_task_output()

        return {"video_url": task.output_url}

    except TaskFailedError as e:
        return JSONResponse(status_code=500, content={"detail": "Video generation failed.", "error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": "Unexpected error during video generation.", "error": str(e)})
