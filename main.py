from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from fastapi.openapi.docs import get_swagger_ui_html
from openai import OpenAI  # 👈 new import for v1+ client

app = FastAPI()

# Initialize OpenAI client
client = OpenAI()  # 👈 automatically uses OPENAI_API_KEY from env

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "Hello from DALL·E!"}

@app.post("/generate")
async def generate_image(data: PromptRequest):
    try:
        response = client.images.generate(
            model="dall-e-3",  # 👈 required for image generation now
            prompt=data.prompt,
            n=1,
            size="1024x1024"
        )
        image_url = response.data[0].url  # 👈 new response format
        return {"image_url": image_url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")
