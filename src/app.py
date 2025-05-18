import streamlit as st
import os
from dotenv import load_dotenv
import subprocess
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import hub
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, List

load_dotenv()

# --- Función para listar modelos Ollama disponibles ---
def listar_modelos_ollama():
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        if result.returncode == 0:
            lineas = result.stdout.strip().split('\n')
            modelos = [line.split()[0] for line in lineas[1:] if line.strip()]
            return modelos
    except FileNotFoundError:
        pass
    return []

# --- Obtener modelos disponibles ---
modelos_ollama = listar_modelos_ollama()

# --- Parámetros UI ---
modelo_seleccionado = st.sidebar.selectbox(
    "Selecciona el modelo Ollama",
    modelos_ollama if modelos_ollama else ["mistral"]
)
temperature = st.sidebar.slider("Temperatura", 0.0, 1.5, 0.7, 0.1)
top_p = st.sidebar.slider("Top-p", 0.0, 1.0, 0.9, 0.05)
top_k = st.sidebar.slider("Top-k", 1, 100, 40, 1)

# --- Inicializar LLM y embeddings ---
llm = ChatOllama(
    model=modelo_seleccionado,
    temperature=temperature,
    num_predict=top_k,
    top_p=top_p,
)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- Vector Store ---
vector_store = Chroma(
    collection_name="mi_coleccion_pdf",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

# --- Procesar PDF ---
def process_pdf_with_langsmith(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)
    vector_store.add_documents(splits)

# --- Prompt predefinido ---
prompt = hub.pull("rlm/rag-prompt")

# --- Estado y pasos LangGraph ---
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=5)
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# --- UI en Streamlit ---
st.title("Chatbot Modulo1")

uploaded_file = st.file_uploader("Sube tu PDF", type="pdf")

if uploaded_file:
    os.makedirs("./data", exist_ok=True)
    save_path = f"./data/{uploaded_file.name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("PDF cargado correctamente")

    if st.button("Procesar PDF"):
        process_pdf_with_langsmith(save_path)
        st.session_state["pdf_procesado"] = True
        st.success("PDF procesado e indexado")

if "pdf_procesado" not in st.session_state:
    st.session_state["pdf_procesado"] = False

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if st.session_state["pdf_procesado"]:
    user_question = st.text_input("Haz una pregunta sobre el PDF")
    if user_question:
        st.session_state.messages.append(("user", user_question))

        # --- Usar LangGraph para obtener la respuesta ---
        response = graph.invoke({"question": user_question})
        answer = response["answer"]

        st.session_state.messages.append(("bot", answer))

    # Mostrar historial
    for role, msg in st.session_state.messages:
        if role == "user":
            st.markdown(f"🧑‍💻 **Tú:** {msg}")
        else:
            st.markdown(f"🤖 **Bot:** {msg}")
