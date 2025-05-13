from langchain_core.chat_history import  BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage


class ConversationSummaryBufferMemory(BaseChatMessageHistory):
    def __init__(self, llm: ChatGoogleGenerativeAI, k: int):
        self.llm = llm
        self.k = k
        self.messages: list[BaseMessage] = []

    def add_messages(self, messages: list[BaseMessage]) -> None:

        existing_summary: BaseMessage | None = None
        old_messages: list[BaseMessage] | None = None

        if len(self.messages) > 0 and isinstance(self.messages[0], SystemMessage):
        #  Eğer eski özet varsa onu alıyoruz burda.
            existing_summary = self.messages.pop(0)

        # Mesajı burda ekliyoruz yenisini
        self.messages.extend(messages)

        #Eğer mesaj fazla ise bir kısmı alıp old_messages a atıyoruz.
        if len(self.messages) > self.k:
            old_messages = self.messages[:self.k]
            self.messages = self.messages[-self.k:]

        # old_messages none ise zaten özetlencek birşey yok
        if old_messages is None:
            return

        #Summary prompt ile old messages varsa onu summary ile birleştiriyoruz.
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "Given the existing conversation summary and the new messages, "
                "generate a new summary of the conversation. Ensuring to maintain "
                "as much relevant information as possible."
            ),
            HumanMessagePromptTemplate.from_template(
                "Existing conversation summary:\n{existing_summary}\n\n"
                "New messages:\n{old_messages}"
            )
        ])

        #new summary burda üretiyoruz llm den.
        new_summary = self.llm.invoke(summary_prompt.format_messages(
            old_messages=old_messages,
            existing_summary=existing_summary
        ))

        #Self messages a ekliyoruz burda da ikisiniede
        self.messages = [SystemMessage(content=new_summary.content)] + self.messages


    def clear(self) -> None:
        #Mesajları siliyoruz burda.
        self.messages = []

    def aget_messages(self) -> list[BaseMessage]:
        return self.messages
