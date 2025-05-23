import asyncio

from langchain.callbacks.base import AsyncCallbackHandler


class QueueCallbackHandler(AsyncCallbackHandler):
    def __init__(self, queue: asyncio.Queue):
        self.queue = queue
        self.final_answer_seen = False

    #Bu fonksiyon aslında her await de çalışır benım kodumdaki.Amacı ise tokenler doldukça onları yield olark döner.
    #Eğer sona geldiysekde ona göre farklı işlemlerde yapar.
    async def __aiter__(self):
        while True:
            if self.queue.empty():
                await asyncio.sleep(0.1)
                continue
            token_or_done = await self.queue.get()
            if token_or_done == "<<DONE>>":
                return
            else:
                yield token_or_done

    #LLM tarafından otomatık olarak bu fonksiyon çalıştırılır.Her tokenı queue ye koyar
    async def on_llm_new_token(self, *args, **kwargs) -> None:
        chunk = kwargs.get("chunk")
        if chunk and chunk.message.additional_kwargs.get("tool_calls"):
            if chunk.message.additional_kwargs["tool_calls"][0]["function"]["name"] == "final_answer":
                self.final_answer_seen = True
        self.queue.put_nowait(kwargs.get("chunk"))

    #LLM işlemlerin sonuna geldini diye kontrol eder.Ona göre done veya step end ile ayort etmemizi sağlar.
    async def on_llm_end(self, *args, **kwargs) -> None:
        if self.final_answer_seen:
            self.queue.put_nowait("<<DONE>>")
        else:
            self.queue.put_nowait("<<STEP_END>>")

