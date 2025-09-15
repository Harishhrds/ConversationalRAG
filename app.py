import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import langchain
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
import time

load_dotenv()

# load the GROQ API KEY
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key,model="Llama3-8b-8192")
from langchain.prompts import ChatPromptTemplate

template = """Answer the question based on the context below only.

<context>
{context}
</context>

Question: {input}
"""

prompts = ChatPromptTemplate.from_template(template)

def create_vector_embeding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers") # Data Ingestion
        st.session_state.docs = st.session_state.loader.load() # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents( st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents( st.session_state.final_documents[:50],st.session_state.embeddings)

user_prompt = st.text_input("Enter your query from research paper")
if st.button("Document Embedding"):
    create_vector_embeding()
    st.write("Vector Database is Ready")

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompts)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({'input':user_prompt})
    print(f"Response_time:{time.process_time()-start}")

    st.write(response['answer'])

    ## with streamlit expander - shows the similarity documents
    with st.expander("Document Similarity Search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("________")




