from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

app = FastAPI()

# Allow only your WordPress site to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.lulati.com"],  # Your WordPress frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PromptRequest(BaseModel):
    prompt: str
    size: str = "512x512"  # default size if not provided

@app.get("/")
def read_root():
    return {"message": "Hello from DALL·E!"}

@app.post("/generate")
async def generate_image(data: PromptRequest):
    valid_sizes = {"256x256", "512x512", "1024x1024"}
    if data.size not in valid_sizes:
        raise HTTPException(status_code=400, detail=f"Invalid size. Must be one of {valid_sizes}")

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
        raise HTTPException(status_code=500, detail=str(e))
