# AI Image Generator API

A simple FastAPI backend for generating images using OpenAI's DALL·E model. Deployable on Render.

## Endpoint

**POST** `/generate`

**Body:**
```json
{ "prompt": "A Black superhero in a futuristic city" }
