import os
from pathlib import Path

import aiohttp
from langchain.tools import  tool
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from Agent.Tools.Article import Article
from pydantic import SecretStr


# Constants and Configuration
GOOGLE_API_KEY = SecretStr(os.environ["GOOGLE_API_KEY"])
SERPAPI_API_KEY = SecretStr(os.environ["SERPAPI_API_KEY"])

@tool
def read_pdf_and_save(path: Path) -> None:
    #pdf yükleme işlemi
    loader = PyPDFLoader(path)
    document = loader.load()

    #pdfi chunklara bölüyoruz verim için.
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(document)

    #Embedding modeli tanımlıyoruz
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    #Chroma ile pdfi kalıcı hale getiriyoruz
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding,
        persist_directory="vectorstore/pdf_store"
    )
    #Kalıcı olarak saklıyoruz
    vectorstore.persist()


@tool
async def serp_api_search(query: str) -> list[Article]:
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
