from typing import Any, Generator

import litellm
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI()


class ResponseBody(BaseModel):
    state: str
    message: str | None = None

    class Config:
        extra = "allow"


def stream(prompt: str) -> Generator[str, Any, None]:
    yield ResponseBody(state="accepted", message=prompt.lower()).model_dump_json()
    accumulated_message = ""
    response: litellm.CustomStreamWrapper = litellm.completion(
        model="ollama/gemma3",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt.lower()},
        ],
        api_base="http://localhost:11434",
        stream=True,
    )
    for chunk in response:
        if chunk.choices[0].finish_reason is not None:
            yield ResponseBody(state="finished", **chunk.model_dump()).model_dump_json()
        else:
            delta_content = chunk.choices[0].delta.content
            accumulated_message += delta_content or ""
            yield ResponseBody(state="streaming", message=accumulated_message).model_dump_json()


@app.post(
    "/",
    responses={
        200: {
            "content": {
                "application/json": {"schema": {"type": "text/event-stream", "items": ResponseBody.model_json_schema()}}
            },
            "description": "Stream of items",
        }
    },
)
def main() -> StreamingResponse:
    PROMPT = "Hello! How are you doing today?"
    return StreamingResponse(content=stream(PROMPT), media_type="text/event-stream")
