from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import openai
import os
import requests
from io import BytesIO
from fastapi.openapi.docs import get_swagger_ui_html

app = FastAPI()

# Use the OpenAI client (new SDK structure)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def read_root():
    return {"message": "Hello from DALL·E!"}

@app.post("/generate")
async def generate_image(data: PromptRequest):
    try:
        response = client.images.generate(
            model="dall-e-3",  # Or "dall-e-2" if needed
            prompt=data.prompt,
            n=1,
            size="1024x1024",
            response_format="url"
        )
        image_url = response.data[0].url

        # Download the image content
        image_response = requests.get(image_url)
        if image_response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch the image.")

        # Wrap in a byte stream for streaming response
        image_stream = BytesIO(image_response.content)

        return StreamingResponse(image_stream, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")
