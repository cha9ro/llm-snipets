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
        temperature=0.9,
    )
    completion = ""
    async for chunk in response:
        completion += chunk.choices[0].delta.content or ""
        if chunk.choices[0].finish_reason != "stop":
            yield f"{json.dumps({"state": "generationg", "content": chunk.choices[0].delta.content})}\n"
        else:
            yield f"{json.dumps({"state": "completed", "content": completion})}\n"
            break


async def main() -> AsyncGenerator[str, None]:
    yield f"{json.dumps({"state": "accepted", "content": "hello world"})}\n"
    async for chunk in acreate():
        yield chunk
    yield f"{json.dumps({"state": "ending", "content": "goodbye world"})}\n"


app = FastAPI()


@app.get("/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(main(), media_type="text/event-stream")
