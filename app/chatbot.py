from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from huggingface_hub import hf_hub_download
import os

def load_chatbot(persist_path="faiss_index"):
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    
    model_name = "microsoft/Phi-3-mini-4k-instruct-gguf"
    model_file = "Phi-3-mini-4k-instruct-q4.gguf"
    
    print("Downloading fast GGUF model...")
    
    model_path = hf_hub_download(
        repo_id=model_name,
        filename=model_file,
        local_dir=os.path.join(os.path.dirname(__file__), '..', 'models'), 
        local_dir_use_symlinks=False
    )
    print("Model download complete.")

    
    local_llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=1,  
        n_batch=512,
        n_ctx=4096,     
        verbose=True
    )

    template = """
       
        <|user|>
        You are a Neuroscience Research Assistant. Use the provided Neuroscience Context 
        to answer the Question accurately. 
        Your answer must be based *only* on the context.
        ...

        If the answer is not in the context, just say "I'm sorry, I couldn't find the answer to that in the provided document."

        Context:
        {context}

        Question:
        {question}

        <|assistant|>
        Answer:
        """
                
    PROMPT = PromptTemplate(
        template=template, input_variables=["context", "question"]
    )

    
    qa_chain = RetrievalQA.from_chain_type(
        llm=local_llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain