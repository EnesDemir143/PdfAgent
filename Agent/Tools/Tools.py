from langchain.tools import  tool
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma


@tool
def read_pdf_and_save():
    #pdf yükleme işlemi
    loader = PyPDFLoader('docs/example.pdf')
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