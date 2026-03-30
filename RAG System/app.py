import streamlit as st
import os
import sys

# Ensure absolute imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.rag.pipeline import RAGPipeline
from src.rag.document_processor import DocumentProcessor
from src.rag.retriever import MedicalRetriever

# Page Config
st.set_page_config(
    page_title="Healthcare Info Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

if "pipeline" not in st.session_state:
    # Ensure vector db exists
    if not os.path.exists("vector_db"):
        with st.spinner("Initializing medical knowledge base..."):
            processor = DocumentProcessor()
            retriever = MedicalRetriever()
            data_dir = os.path.join(os.getcwd(), "data")
            if os.path.exists(data_dir):
                chunks = processor.load_and_split(data_dir)
                retriever.create_vector_store(chunks)
    st.session_state.pipeline = RAGPipeline()

# Sidebar
with st.sidebar:
    st.title("Settings")
    st.info("This AI answers healthcare questions using verified hospital FAQs and clinical guidelines. It detects emergencies and highlights sources.")
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main UI
st.title("🏥 Healthcare Information Assistant")
st.markdown("---")

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption(f"Sources: {', '.join(message['sources'])}")

# User Input
if prompt := st.chat_input("Ask a medical question (e.g., 'What are hypertension symptoms?')"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Consulting sources..."):
            answer, sources = st.session_state.pipeline.process_query(prompt)
            st.markdown(answer)
            if sources:
                st.caption(f"Sources: {', '.join(sources)}")
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": sources
            })

# Disclaimer in Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer:** This is an AI-powered educational tool. Not a substitute for professional medical advice. Always consult a doctor for diagnosis or treatment.")
