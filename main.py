from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = FastAPI()

# CORS settings — adjust your frontend URL here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.lulati.com"],  # Change to your frontend domain or use ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment")

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Request model
class PromptRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"  # Default size

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Image Generator API"}

@app.post("/generate-image")
async def generate_image(data: PromptRequest):
    if not data.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

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
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
