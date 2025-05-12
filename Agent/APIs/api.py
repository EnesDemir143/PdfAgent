import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from Agent.Agent import agent_executor
from Agent.StreamingHandler import QueueCallbackHandler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

async def token_generate(content: str, streamer: QueueCallbackHandler):
    task = asyncio.create_task(agent_executor.invoke(
        input=content,
        streamer=streamer,
        verbose=True
    ))

    async for token in streamer:
        try:
            if token == "<<STEP_END>>":
                yield '</step>'
            elif tool_calls := token.message.additional_kwargs.get('tool_calls'):
                if tool_name := tool_calls[0]['function']['name']:
                    yield f"<step><step_name>{tool_name}</step_name>"
            if tool_args := tool_calls[0]["function"]["arguments"]:
                yield tool_args
        except Exception as e:
            print(f"Error streaming token: {e}")
            continue
    await task

@app.post('/invoke')
async def invoke(content: str):
    queue : asyncio.Queue = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)

    return StreamingResponse(
        token_generate(content, streamer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )