from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os
from fastapi.openapi.docs import get_swagger_ui_html

app = FastAPI()

# Make sure this env var is set!
openai.api_key = os.getenv("OPENAI_API_KEY")

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "Hello from DALL·E!"}

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

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")
