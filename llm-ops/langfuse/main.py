from dotenv import load_dotenv
from langfuse.decorators import observe  # type: ignore
from langfuse.openai import OpenAI  # type: ignore # OpenAI integration

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")


@observe()
def story():
    return (
        client.chat.completions.create(
            model="llama3.1",
            max_tokens=100,
            messages=[
                {"role": "system", "content": "You are a great storyteller."},
                {"role": "user", "content": "Once upon a time in a galaxy far, far away..."},
            ],
        )
        .choices[0]
        .message.content
    )


@observe()
def main():
    print(story())


if __name__ == "__main__":
    load_dotenv()
    main()
