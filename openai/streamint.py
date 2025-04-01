from typing import Any, Generator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from openai import OpenAI

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

app = FastAPI()


class ResponseBody(BaseModel):
    state: str
    message: str | None = None

    class Config:
        extra = "allow"


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

    def stream() -> Generator[str, Any, None]:
        yield ResponseBody(state="accepted", message=PROMPT.lower()).model_dump_json()
        accumulated_message = ""
        response = client.chat.completions.create(
            model="gemma3",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": PROMPT.lower()},
            ],
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].finish_reason is not None:
                yield ResponseBody(state="finished", **chunk.model_dump()).model_dump_json()
            else:
                delta_content = chunk.choices[0].delta.content
                accumulated_message += delta_content or ""
                yield ResponseBody(state="streaming", message=accumulated_message).model_dump_json()

    return StreamingResponse(content=stream(), media_type="text/event-stream")
