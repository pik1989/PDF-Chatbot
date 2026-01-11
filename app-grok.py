import streamlit as st
from langchain_xai import ChatXAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
import os

# ───────────────────────────────────────────────────────────────
# CONFIGURATION
# ───────────────────────────────────────────────────────────────

PDF_PATH = "Research Paper - Final.pdf"           # ← Change if needed
MODEL_NAME = "grok-4"                             # or "grok-beta" etc.

# You can set this in .env / secrets / environment variable
# For demo we use text input (not recommended for production)
DEFAULT_API_KEY = ""   # ← leave empty

# ───────────────────────────────────────────────────────────────
# PROMPT TEMPLATE
# ───────────────────────────────────────────────────────────────

RAG_PROMPT = """
You are an intelligent assistant trained to understand and provide information based on the context provided.
Use **only** the following context to answer the question.
If you don't know the answer or the information is not in the context - say so.

Context:
{context}

Question: {question}

Answer concisely and clearly:
"""

# ───────────────────────────────────────────────────────────────
# CACHE DOCUMENT LOADING & CONTEXT CREATION
# ───────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading and processing PDF...")
def load_pdf_context(pdf_path):
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found at: {pdf_path}")
        st.stop()
        
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        context = "\n\n".join(doc.page_content for doc in documents)
        return context
    except Exception as e:
        st.error(f"Error while loading PDF: {str(e)}")
        st.stop()

# ───────────────────────────────────────────────────────────────
# STREAMLIT APP
# ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="PDF Q&A with Grok", page_icon="📄")

st.title("PDF Question Answering with Grok")
st.caption(f"Model: {MODEL_NAME}  •  Document: {os.path.basename(PDF_PATH)}")

# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    
    api_key = st.text_input(
        "xAI API Key",
        value=DEFAULT_API_KEY,
        type="password",
        help="Get your key at https://console.x.ai"
    )
    
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.1)
    
    if st.button("Clear Chat", type="primary"):
        st.session_state.messages = []
        st.rerun()

# ── Main content ──────────────────────────────────────────────────

# Load document only once
context = load_pdf_context(PDF_PATH)
st.success("PDF loaded successfully ✓", icon="✅")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask anything about the research paper..."):
    
    if not api_key.strip():
        st.error("Please provide your xAI API key in the sidebar", icon="🔑")
        st.stop()
        
    # Add user message to chat history & display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare full prompt
    full_prompt = PromptTemplate.from_template(RAG_PROMPT).format(
        context=context,
        question=prompt
    )

    # Create model instance
    try:
        chat = ChatXAI(
            model=MODEL_NAME,
            api_key=api_key,
            temperature=temperature
        )
    except Exception as e:
        st.error(f"Failed to initialize Grok model: {str(e)}")
        st.stop()

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = chat.invoke(full_prompt)
                answer = response.content
                
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as e:
                st.error(f"API Error: {str(e)}")