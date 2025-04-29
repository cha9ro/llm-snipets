import json
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from openai import AsyncOpenAI


async def acreate() -> AsyncGenerator[str, None]:
    async_client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="dummy")
    response = await async_client.chat.completions.create(
        model="gemma3",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello! How are you?"},
        ],
        stream=True,
    )
    async for chunk in response:
        yield json.dumps({"state": "generationg", "content": chunk.choices[0].delta.content})


async def main() -> AsyncGenerator[str, None]:
    yield json.dumps({"state": "accepted", "content": "hello world"})
    async for chunk in acreate():
        yield chunk
    yield json.dumps({"state": "completed", "content": "goodbye world"})


app = FastAPI()


@app.get("/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(main(), media_type="text/event-stream")
