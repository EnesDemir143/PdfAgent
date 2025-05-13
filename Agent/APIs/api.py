import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import asyncio

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from Agent.Agent import agent_executor
from Agent.StreamingHandler import QueueCallbackHandler
from pydantic import BaseModel

os.makedirs("temp_pdfs", exist_ok=True)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

async def token_generator(content: str, streamer: QueueCallbackHandler):
    task = asyncio.create_task(agent_executor.invoke(
        input=content,
        streamer=streamer,
        verbose=True  # set to True to see verbose output in console
    ))
    # initialize various components to stream
    async for token in streamer:
        try:
            if token == "<<STEP_END>>":
                # send end of step token
                yield "</step>"
            elif tool_calls := token.message.additional_kwargs.get("tool_calls"):
                if tool_name := tool_calls[0]["function"]["name"]:
                    # send start of step token followed by step name tokens
                    yield f"<step><step_name>{tool_name}</step_name>"
                if tool_args := tool_calls[0]["function"]["arguments"]:
                    # tool args are streamed directly, ensure it's properly encoded
                    yield tool_args
        except Exception as e:
            print(f"Error streaming token: {e}")
            continue
    await task


from fastapi import UploadFile, File, Form
@app.post("/invoke")
async def invoke(content: str = Form(...), pdf_file: UploadFile = File(None)):
    queue: asyncio.Queue = asyncio.Queue()
    streamer = QueueCallbackHandler(queue)

    if pdf_file is not None:
        temp_path = f"temp_pdfs/{pdf_file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await pdf_file.read())
        content += f"\n\n[PDF_PATH:{temp_path}]"

    return StreamingResponse(
        token_generator(content, streamer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )