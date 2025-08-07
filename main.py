# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import os

# Initialize FastAPI app
app = FastAPI()

# Set your OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the request model
class PromptRequest(BaseModel):
    prompt: str

# Endpoint to generate the image based on the prompt
@app.post("/generate")
async def generate_image(data: PromptRequest):
    try:
        # Call the OpenAI API to generate an image
        response = openai.Image.create(
            prompt=data.prompt,
            n=1,
            size="1024x1024"
        )
        # Extract the URL of the generated image
        image_url = response['data'][0]['url']
        return {"image_url": image_url}
    except Exception as e:
        # If there's an error, raise an HTTP exception
        raise HTTPException(status_cod_
