import anyio
from claude_code_sdk import query, ClaudeCodeOptions, Message
from pathlib import Path

async def main():
    messages: list[Message] = []
    
    async for message in query(
        prompt="Provide a list of all code files related to workorders. Make sure the output is valid JSON, in the form of [\"path/To/File1\", \"path/To/File2\"]",
        options=ClaudeCodeOptions(max_turns=3, cwd="/Users/uribermankleiner/repos/maintainx", model="claude-3-7-sonnet-20250219")
    ):
        messages.append(message)
    
    print(messages)

anyio.run(main)