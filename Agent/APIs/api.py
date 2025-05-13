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

import json
async def token_generate(content: str, streamer: QueueCallbackHandler):
    task = asyncio.create_task(agent_executor.invoke(
        input=content,
        streamer=streamer,
        verbose=True
    ))

    async for token in streamer:
        try:
            if token == "<<STEP_END>>":
                yield "</step>"

            elif tool_calls := getattr(token.message, "tool_calls", None):
                if isinstance(tool_calls, list) and tool_calls:
                    tool_name = tool_calls[0].get("name")
                    if tool_name:
                        yield f"<step><step_name>{tool_name}</step_name>"

                    tool_args = tool_calls[0].get("args", {})

                    # final_answer için JSON stringi değil, XML etiketleri ile detayları verelim
                    if tool_name == "final_answer":
                        answer = tool_args.get("answer", "")
                        tools_used = tool_args.get("tools_used", [])
                        yield f"<answer>{answer}</answer>"
                        yield f"<tools_used>{json.dumps(tools_used)}</tools_used>"

                        # final_answer geldi, stream durduruluyor:
                        print("final_answer alındı, streaming durduruluyor.")
                        break  # veya return ile fonksiyonu sonlandırabilirsiniz

                    else:
                        # Diğer tool argümanları JSON string olarak gönderilebilir
                        if tool_args:
                            yield json.dumps(tool_args)

            elif hasattr(token, "message") and token.message.content:
                yield token.message.content

        except Exception as e:
            print(f"Error streaming token: {e}")
            continue

    # Streaming tamamlandığında task sonucu alınıyor ama tekrar yield etmiyoruz
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
        token_generate(content, streamer),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )