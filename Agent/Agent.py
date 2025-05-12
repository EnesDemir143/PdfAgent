import os
from gc import callbacks

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
from Agent.Memory.memory import ConversationSummaryBufferMemory
import Tools.Tools as tools
from Agent.StreamingHandler import QueueCallbackHandler
from Agent.Tools.Tools import read_pdf_and_save, serp_api_search

model = 'models/gemini-2.5-flash-preview-04-17'
GOOGLE_API_KEY = SecretStr(os.environ['GOOGLE_API_KEY'])

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
        self.tools = [read_pdf_and_save, serp_api_search]
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

    async def stream(self, query: str, stramer: QueueCallbackHandler) -> AIMessage:
        response = self.agent.with_config(
            callbacks=[stramer]
        )

    async def invoke(self, input: str, streamer: QueueCallbackHandler, verbose: bool = False):
        count = 0
        final_answer: str | None = None
        agent_scratchpad: list[AIMessage | ToolMessage] = []
