from langchain_google_genai import ChatGoogleGenerativeAI

from Agent.Memory.memory import ConversationSummaryBufferMemory

chat_map = {}
def get_chat_history(session_id: str, llm: ChatGoogleGenerativeAI, k:int):
    if session_id not in chat_map:
        chat_map[session_id]= ConversationSummaryBufferMemory(llm=llm, k=k)
    return chat_map[session_id]

