import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

VECTOR_STORE_PATH = "faiss_index"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_LLM = "mistral:latest"

# --- Procesar PDF y construir índice FAISS ---
def process_pdf(file_path: str):
    print("📄 Cargando documento...")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = splitter.split_documents(documents)
    print(f"🔍 Se cargaron {len(docs)} fragmentos. Generando embeddings...")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vectordb = FAISS.from_documents(docs, embeddings)

    print("💾 Guardando índice FAISS...")
    vectordb.save_local(VECTOR_STORE_PATH)

    print(f"✅ {len(docs)} fragmentos indexados en FAISS.")

# --- Cargar índice FAISS desde disco ---
def get_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    return FAISS.load_local(VECTOR_STORE_PATH, embeddings)

# --- Cargar cadena QA ---
def load_qa_chain(model_name=DEFAULT_LLM, temperature=0.7, top_p=0.9, top_k=40):
    vectordb = get_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    llm = Ollama(
        model=model_name,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain

# --- Responder pregunta ---
def answer_question(query: str, model_name=DEFAULT_LLM, temperature=0.7, top_p=0.9, top_k=40) -> str:
    qa_chain = load_qa_chain(model_name, temperature, top_p, top_k)
    result = qa_chain({"query": query})

    print("📚 Documentos fuente utilizados:")
    for doc in result["source_documents"]:
        print(f"\n📝 [Fuente] {doc.metadata.get('source', 'Sin metadata')}")
        print(doc.page_content[:500], "...")  # muestra los primeros 500 caracteres

    return result["result"]
