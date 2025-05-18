from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
import os

CHROMA_PATH = "chroma_db"

def process_pdf(file_path: str):
    # 1. Cargar el PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # 2. Dividir el texto en fragmentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # 3. Generar embeddings con Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Guardar embeddings en Chroma
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    vectordb.persist()

    print(f"{len(docs)} fragmentos indexados correctamente.")

def get_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
