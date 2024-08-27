import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI()

# openai.api_key = "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"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GPTRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


@app.post("/gpt4/")
async def get_gpt4_response(request: GPTRequest):
    try:

        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": request.prompt}],
            max_tokens=request.max_tokens
        )

        return {"response": response.choices[0].message['content']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
