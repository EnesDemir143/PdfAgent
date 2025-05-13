import asyncio
import os

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from pydantic import SecretStr
from langchain_openai import ChatOpenAI

from Agent.Memory.memory import ConversationSummaryBufferMemory
from Agent.StreamingHandler import QueueCallbackHandler
from Agent.Tools.Tools import read_pdf_and_save, final_answer, serp_api_search, query_pdf_store, think_through_question

#Model tanımlaması
model = 'gpt-4o-mini'
#API yı alıyoruz .env den.
OPENAI_API_KEY = SecretStr(os.environ['OPENAI_API_KEY'])

#LLM tanımlaması
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.0,
    streaming=True,
    api_key=OPENAI_API_KEY
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
        "If the user uploads or refers to a PDF, you must first call the `read_pdf_and_save` tool, then use `query_pdf_store` to extract information from it before responding. Do not answer the user directly — only use the `final_answer` tool after the relevant tools have been called."
        "Answer general questions and engage in conversations naturally. "
        "If you call the `think_through_question` tool, you must then follow its reasoning and call the necessary tools it suggests."
        "After using all needed tools, you must summarize everything with the `final_answer` tool. Do NOT stop after thinking. "
        """ Example: 
        User: "Who is Einstein and what is the capital of France?"
        → Call think_through_question
        → Then call serp_api_search twice
        → Then call final_answer"""
        "Always use the final_answer tool to provide the final response in the format {{'answer': '...', 'tools_used': [...]}}."
        "You MUST use the `final_answer` tool to give your final reply. Do NOT reply with plain text. Responses without calling this tool will be ignored."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

class Agent:
    def __init__(self, max_iter=3):
        tools = [final_answer, read_pdf_and_save, query_pdf_store]
        self.tools = {tool.name: tool.coroutine for tool in tools}
        self.max_iter = max_iter
        self.chat_history = ConversationSummaryBufferMemory(llm=llm, k=self.max_iter)
        self.agent = (
            {
                'input': lambda x: x['input'],
                'agent_scratchpad': lambda x: x.get("agent_scratchpad", []),
                'chat_history': lambda x: x['chat_history']
            }
            | prompt
            | llm.bind_tools(tools, tool_choice='any')
        )

    async def execute_tool(self, tool_call: AIMessage) -> ToolMessage:
        print('tool_call', tool_call)
        if tool_call.tool_calls:
            tool_name = tool_call.tool_calls[0]["name"]
            tool_args = tool_call.tool_calls[0]["args"]
            tool_out = await self.tools[tool_name](**tool_args)
        else:
            return None
        return ToolMessage(
            content=f"{tool_out}",
            tool_call_id=tool_call.tool_calls[0]["id"]
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
            "chat_history":  self.chat_history.aget_messages(),
            "agent_scratchpad": agent_scratchpad
        }):
            #Burda token sonucu olan argumanları alıyoruz.Bunlar fonksiyonun id'side olabilir.Parametreleride.
            tool_calls = token.tool_calls
            if tool_calls:
                #Eğer fonksiyon id si ise ekliyoruz.
                if tool_calls[0]['id']:
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
            ) for x in outputs
        ]



    async def invoke(self, input: str, streamer: QueueCallbackHandler, verbose: bool = False):
        count = 0
        final_answer: str | None = None
        agent_scratchpad: list[AIMessage | ToolMessage] = getattr(self, 'agent_scratchpad', [])

        while count< self.max_iter:
            tool_calls = await self.stream(query=input, stramer=streamer, agent_scratchpad=agent_scratchpad)

            tool_executes = await asyncio.gather(
                *[self.execute_tool(tool_call) for tool_call in tool_calls]
            )
            if tool_executes:
                break
            print('tool_execute bitti')
            id2tool_executed = {tool_call.tool_call_id: tool_execute for tool_call, tool_execute in
                                zip(tool_calls, tool_executes)}

            for tool_call in tool_calls:
                tool_call_id = tool_call.tool_call_id
                if tool_call_id in id2tool_executed:
                    agent_scratchpad.extend([tool_call, id2tool_executed[tool_call_id]])
                else:
                    print(f"Warning: tool_call_id {tool_call_id} not found or execution failed. {tool_call}")
            print('agent_scratchpad eklemesi tamamlandı.')
            count+=1
            #Burda tooların hepsindn final varmı bakılır.Varsa eğer onun answer kısmını alıyoruz.

            found_final_answer = False
            for tool_call in tool_calls:
                if tool_call.tool_calls[0]["name"] == "final_answer":
                    final_answer_call = tool_call.tool_calls[0]
                    final_answer = final_answer_call["args"]["answer"]
                    found_final_answer = True
                    break
                    print('final answer check tamamlandı.')

            if found_final_answer:
                break

        #Custom olan kendi history'ye eklemeler yapıyorum burda.
        self.chat_history.add_messages([
            HumanMessage(content=input),
            AIMessage(content=final_answer if final_answer else 'No answer found')
        ])

        return final_answer_call if final_answer else {"answer": "No answer found", "tools_used": []}

agent_executor = Agent()