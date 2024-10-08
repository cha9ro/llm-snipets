from dotenv import load_dotenv

import openai
from langsmith import traceable
from langsmith.wrappers import wrap_openai

client = wrap_openai(client=openai.OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"))


@traceable
def pipeline(user_input: str) -> str | None:
    result = client.chat.completions.create(messages=[{"role": "user", "content": user_input}], model="llama3.1")
    return result.choices[0].message.content


if __name__ == "__main__":
    load_dotenv()
    result = pipeline("Hello world!")
    print(result)
