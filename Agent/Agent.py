import asyncio
import os
from gc import callbacks

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import tool_call
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from Agent.Memory.memory import ConversationSummaryBufferMemory
import Tools.Tools as tools
from Agent.StreamingHandler import QueueCallbackHandler
from Agent.Tools.Tools import read_pdf_and_save, serp_api_search

#Model tanımlaması
model = 'models/geini-2.5-flash-preview-04-17'
#API yı alıyoruz .env den.
GOOGLE_API_KEY = SecretStr(os.environ['GOOGLE_API_KEY'])

#LLM tanımlaması
llm = ChatGoogleGenerativeAI(
    google_api_key=GOOGLE_API_KEY,
    model=model,
    tempreture=0.0
).configurable_fields(
    callbacks=ConfigurableField(
        id="callbacks",
        name="callbacks",
        description="A list of callbacks to use for streaming",
    )
)

#İstediğimiz şeyleri yapması için system prompt ile yönlendirme yapıyoruz.
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful and intelligent assistant. "
        "You are skilled at summarizing and commenting on PDF documents. "
        "When given a PDF, extract the key points, create a clear and concise summary, and provide insightful comments or critiques. "
        "You are also capable of conducting internet research when needed, to enrich your responses with accurate, up-to-date, and relevant information. "
        "For technical or academic documents, ensure accuracy, depth, and critical thinking in your summaries and comments."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

class Agent:
    def __init__(self, max_iter=3):
        tools = [read_pdf_and_save, serp_api_search]
        self.tools = {tool.name:tool.coroutine for tool in tools}
        self.chat_history = ConversationSummaryBufferMemory()
        self.max_iter = max_iter
        self.agent = (
            {
                'input': lambda x: x['input'],
                'agent_scratchpad': lambda x: x.get("agent_scratchpad", []),
                'chat_history': lambda x: x['chat_history']
            }
            | prompt
            | llm.bind_tools(self.tools, tool_choice='any')
        )
    #Bu fonksiyon aslında streaming ile gelen tokenları işliyor ve birleştiriyor.Yani tool ve parametrekleri kısaca
    async def stream(self, query: str, stramer: QueueCallbackHandler, agent_scratchpad: list[AIMessage | ToolMessage]) -> list[AIMessage]:
        #Burda straming yapmak için gereken confiiği alıyorum llmden
        response = self.agent.with_config(
            callbacks=[stramer]
        )
        #Burda her responseyi streaming olarak almamızı saglıyor bu.
        outputs = []
        async for token in response.astream({
            "input": query,
            "chat_history": self.chat_history,
            "agent_scratchpad": agent_scratchpad
        }):
            #Burda token sonucu olan argumanları alıyoruz.Bunlar fonksiyonun id'side olabilir.Parametreleride.
            tool_calls = token.additional_kwargs.get('tool_calls')
            if tool_calls:
                #Eğer fonksiyon id si ise ekliyoruz.
                if tool_calls[0][id]:
                    outputs.append(token)
                #Aynı fonksiyon ise topluyoruz surekli bu da bize bir tool sonucu oluyor elimizde.İdsi olan ve parametreleri olan bir mesaj.
                else:
                    outputs[-1]+=token
            else:
                pass
        #Sonra burda her toolu argumanlarınıda veriyoruz idsinide.Böylece aslında her toolu ve onun parametreleri ayırt ediliyor.
        # Sonrada tool çalıştırılıyor.AI message sayesinde.
        return [
            AIMessage(
                content=x.content,
                tool_call=x.tool_calls,
                tool_call_id=x.tool_calls[0]['id']
            )for x in outputs
        ]



    async def invoke(self, input: str, streamer: QueueCallbackHandler, verbose: bool = False):
        count = 0
        final_answer: str | None = None
        agent_scratchpad: list[AIMessage | ToolMessage] = []

        tool_calls = await self.stream(query=input, stramer=streamer, agent_scratchpad=agent_scratchpad)

        tool_execute = await asyncio.gather(
            *[ece]
        )