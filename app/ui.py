import streamlit as st
from app.chatbot import load_chatbot

st.set_page_config(page_title=" LLM Chatbot ", layout="wide")
st.title("Neuro-Knowledge Assistant")
qa_chain = load_chatbot()
query = st.text_input("what's your query? :) ")
if query:
    with st.spinner("wait boss I am searching, I am not as smart as you are ;) ..."):
        result = qa_chain(query)
        st.markdown(f"**Answer:** {result['result']}")
        st.markdown("Source Chunks")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}:** {doc.page_content[:300]}...")
