from agents import Agent, RunConfig, Runner, trace
from agents.mcp import MCPServer, MCPServerStdio
from dotenv import load_dotenv


async def run(server: MCPServer, prompt: str):
    weather_agent = Agent(
        name="Weather Assistant",
        instructions="Use the tools to answer the user's questions about the weather.",
        mcp_servers=[server],
    )
    result = await Runner.run(starting_agent=weather_agent, input=prompt, run_config=RunConfig(model="o4-mini"))
    print(result.final_output)


async def main(prompt: str):
    load_dotenv()
    async with MCPServerStdio(cache_tools_list=True, params={"command": "uv", "args": ["run", "weather.py"]}) as server:
        with trace(workflow_name="weather agent"):
            await run(server, prompt)


if __name__ == "__main__":
    import asyncio
    import sys

    args = sys.argv[1:]
    asyncio.run(main(args[0] if args else "シアトルの天気は？"))
