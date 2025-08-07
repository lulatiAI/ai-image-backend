from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

app = FastAPI()

# Allow only your WordPress site
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.lulati.com"],  # exact domain match
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PromptRequest(BaseModel):
    prompt: str
    size: str = "1024x1024"  # default size

@app.get("/")
def read_root():
    return {"message": "Hello from DALL·E!"}

@app.post("/generate")
async def generate_image(data: PromptRequest):
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
