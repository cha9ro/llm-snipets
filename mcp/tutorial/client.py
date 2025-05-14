import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters, stdio_client
from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam

load_dotenv()  # load environment variables from .env

MODEL = "gpt-4o"
MAX_TOKENS = 1000
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY environment variable not set. Please set it in your .env file.")


class MCPClient:
    def __init__(self):
        self.session: ClientSession | None = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(api_key=API_KEY)

    async def connect_to_server(self, server_script_path: str):
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(command=command, args=[server_script_path], env=None)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        assert self.session is not None, "Failed to create session"
        await self.session.initialize()

        response = await self.session.list_tools()
        tools = response.tools
        print("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        if self.session is None:
            raise RuntimeError("Session is not initialized. Call connect_to_server() first.")

        tools_res = await self.session.list_tools()
        available_tools: list[ChatCompletionToolParam] = [
            {
                "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.inputSchema},
                "type": "function",
            }
            for tool in tools_res.tools
        ]
        messages: list[ChatCompletionMessageParam] = [
            {"role": "user", "content": query},
        ]

        while True:
            response = self.openai.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=messages,
                tools=available_tools,
                tool_choice="auto",
            )

            msg = response.choices[0].message
            if msg.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
                    }
                )

                for tool_call in msg.tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    tool_response = await self.session.call_tool(tool_name, tool_args)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps([str(c) for c in tool_response.content]),
                        }
                    )
            else:
                messages.append({"role": "assistant", "content": msg.content})
                return msg.content or "(no response)"

    async def chat_loop(self):
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                print("\n" + response)
            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
