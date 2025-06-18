import anyio
from claude_code_sdk import query, ClaudeCodeOptions, Message

async def main():
    messages: list[Message] = []
    
    async for message in query(
        prompt="Write a haiku about app.ts",
        options=ClaudeCodeOptions(max_turns=3, cwd="/Users/uribermankleiner/repos/maintainx")
    ):
        messages.append(message)
    
    print(messages)

anyio.run(main)