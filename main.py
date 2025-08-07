from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "Hello from DALL.E!"}

@app.post("/generate")
async def generate_image(data: PromptRequest):
    try:
        response = openai.Image.create(
            prompt=data.prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response['data'][0]['url']
        return {"image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
