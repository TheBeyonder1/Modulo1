from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama


import os

CHROMA_PATH = "chroma_db"

def process_pdf(file_path: str):
    print(" Cargando documento...")
    # 1. Cargar el PDF
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    

    # 2. Dividir el texto en fragmentos
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    print(f" Se cargaron {len(docs)} páginas. Procesando embeddings...")
    # 3. Generar embeddings con Ollama
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # 4. Guardar embeddings en Chroma
    vectordb = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    vectordb.persist()

    print(f"{len(docs)} fragmentos indexados correctamente.")

def get_vectorstore():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

def load_qa_chain():
    # Cargar embeddings y base vectorial
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Cargar LLM de Ollama
    llm = Ollama(model="llama3.2:3b")  # o mistral, gemma, etc.

    # Crear chain de QA con retrieval
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True
    )

    return qa_chain

def answer_question(query: str) -> str:
    qa_chain = load_qa_chain()
    result = qa_chain(query)

    print("Documentos fuente utilizados:")
    for doc in result["source_documents"]:
        print(f"\n[Fuente] {doc.metadata.get('source', 'Sin metadata')}")
        print(doc.page_content)

    return result["result"]

