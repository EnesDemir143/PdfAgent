from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI


class ConversationSummaryBufferMemory(BaseChatMessageHistory):
    def __init__(self, llm: ChatOpenAI, k: int):
        self.llm = llm
        self.k = k
        self.messages: list[BaseMessage] = []

    def add_messages(self, messages: list[BaseMessage]) -> None:
        existing_summary: str = ""
        old_messages: list[BaseMessage] | None = None

        # Check if we have an existing summary
        if len(self.messages) > 0 and isinstance(self.messages[0], SystemMessage):
            summary_message = self.messages.pop(0)
            existing_summary = summary_message.content

        # Add new messages
        self.messages.extend(messages)

        # Check if we need to summarize
        if len(self.messages) > self.k:
            old_messages = self.messages[:-self.k]  # Messages to summarize
            self.messages = self.messages[-self.k:]  # Keep recent messages
        
            # Prepare messages to be summarized in a readable format
            messages_to_summarize = "\n".join([
                f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}" 
                for msg in old_messages
            ])

            # Create summary prompt
            summary_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(
                    "Given the existing conversation summary and the new messages, "
                    "generate a new summary of the conversation. Ensuring to maintain "
                    "as much relevant information as possible."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Existing conversation summary:\n{existing_summary}\n\n"
                    "New messages:\n{messages_to_summarize}"
                )
            ])

            # Generate new summary
            new_summary = self.llm.invoke(summary_prompt.format_messages(
                existing_summary=existing_summary,
                messages_to_summarize=messages_to_summarize
            ))

            # Add summary as system message at the beginning
            self.messages = [SystemMessage(content=new_summary.content)] + self.messages

    def clear(self) -> None:
        self.messages = []

    def get_messages(self) -> list[BaseMessage]:
        return self.messages
