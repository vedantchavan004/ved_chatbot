import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

FAISS_DB_PATH = "db_faiss"

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(FAISS_DB_PATH, get_embedding_model(), allow_dangerous_deserialization=True)

faiss_db = load_vectorstore()

llm_model = ChatGroq(model="deepseek-r1-distill-llama-70b") 


def retrieve_docs(query):
    return faiss_db.similarity_search(query, k=5)

def get_context(documents):
    return "\n\n".join([doc.page_content for doc in documents])

# Custom prompt template
custom_prompt_template = """
Use the pieces of information provided in the profile to answer the user's question.
You will act as Vedant's assistant, FAQ chatbot, and answer the user's questions based on Vedant.
If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
Don't provide anything out of the given profile and only answer the questions based on the input.
Question: {question} 
Context: {context} 
Answer:
"""

def answer_query(query):
    documents = retrieve_docs(query)
    context = get_context(documents)
    prompt = ChatPromptTemplate.from_template(custom_prompt_template)
    chain = prompt | llm_model
    response = chain.invoke({"question": query, "context": context})
    return response.content.split("</think>")[-1].strip()

# Streamlit UI
st.title("Vedant's FAQ Chatbot")
st.write("Ask a question related to Vedant and his projects.")

query = st.text_input("Enter your question:", "")

if query:
    response = answer_query(query)
    st.write("### Answer:")
    st.write(response)
