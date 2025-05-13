from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.tools import tool
from pathlib import Path
import aiohttp
import os
from pydantic import SecretStr
from Agent.Tools.Article import Article
from langchain.text_splitter import CharacterTextSplitter

GOOGLE_API_KEY = SecretStr(os.environ["GOOGLE_API_KEY"])
SERPAPI_API_KEY = SecretStr(os.environ["SERPAPI_API_KEY"])

@tool
def read_pdf_and_save(path: Path) -> str:
    """
    PDF dosyasını okur, parçalara böler, vektörleştirir ve Chroma'ya kaydeder.
    """
    try:
        print(f"PDF işleme başladı: {path}")
        loader = PyPDFLoader(path)
        document = loader.load()
        print(f"PDF sayfa sayısı: {len(document)}")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(document)
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory="vectorstore/pdf_store"
        )
        vectorstore.persist()
        print("PDF başarıyla işlendi ve kaydedildi.")
        return "PDF başarıyla işlendi."
    except Exception as e:
        print(f"PDF işleme hatası: {e}")
        return f"PDF işleme sırasında hata oluştu: {e}"
@tool
async def serp_api_search(query: str) -> list[Article]:
    """
    Google'da verilen sorgu ile arama yapar ve sonuçları Article listesi olarak döndürür.
    """
    params = {
        'api_key': SERPAPI_API_KEY.get_secret_value(),
        'engine': 'google',
        'q': query
    }
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://serpapi.com/search",
            params=params
        ) as response:
            results = await response.json()
    return [Article.from_serpapi_result(result) for result in results['organic_results']]

@tool
async def final_answer(answer: str, tools_used: list[str]) -> dict[str, str | list[str]]:
    """Use this tool to provide a final answer to the user."""
    return {"answer": answer, "tools_used": tools_used}

@tool
def query_pdf_store(query: str) -> str:
    """
    Kaydedilmiş PDF vektör veritabanında verilen sorguya (query) göre arama yapar ve
    en alakalı doküman parçalarını getirir.
                                        """
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory="vectorstore/pdf_store", embedding_function=embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    combined_text = "\n".join([doc.page_content for doc in docs])
    return combined_text
