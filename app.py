import streamlit as st
import os

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

from langchain.chains import create_retrieval_chain


from dotenv import load_dotenv
load_dotenv()

## load Groq API
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
# one more step
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key= groq_api_key, model_name="llama-3.1-8b-instant")

# prompt
prompt = ChatPromptTemplate.from_template(
"""
You are a helpful assistant.

Use the context to answer the question in a clear, detailed and well-explained way.
If the context is small, expand the explanation using your understanding.

<context>
{context}
</context>

Question: {input}

Answer in a detailed and helpful manner:
"""
)

# loading PDFs

loader = PyPDFDirectoryLoader("research_papers")

docs = loader.load()

# Split text

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

documents = splitter.split_documents(docs)

#Embeddings
if "vectors" not in st.session_state:
    with st.spinner("Embeddings are being created.."):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        # Vectorstore
        vectorstore = FAISS.from_documents(documents,embeddings)

        st.session_state.vectors= vectorstore


st.success("Embeddings created successfully")


# retriver

retriver = st.session_state.vectors.as_retriever()

# Chain

document_chain = create_stuff_documents_chain(llm,prompt)

retrieval_chain = create_retrieval_chain(retriver, document_chain)

## Streamlit UI

st.title("Simple Rag App")

question = st.text_input(
    "Ask a question"
)

# Ask question
import time

if question:
    start = time.process_time()

    response = retrieval_chain.invoke({"input":question})

    st.write(response['answer'])

    st.write(f"Response time : {time.process_time() -start:2f} sec")


    # retrieved chunks

    with st.expander("Document Chunks"):

        for doc in response["context"]:
            st.write(doc.page_content)
            st.write("---")