from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
import os
import pickle

# ======================== Load Data ==========================
# Load text file using TextLoader
TEXT_FILE_PATH = './textfile.txt'

if not os.path.exists(TEXT_FILE_PATH):
    st.error("Text file not found. Ensure 'textfile.txt' is in the correct directory.")
    raise FileNotFoundError(f"File not found: {TEXT_FILE_PATH}")

load_text = TextLoader(TEXT_FILE_PATH, encoding='utf-8')
text_loader = load_text.lazy_load()
text_pages = list(text_loader)  # Convert generator to list

if not text_pages:
    st.error("No data loaded from the text file. Check its content.")
    raise ValueError("Text file appears to be empty or improperly formatted.")

# ================== Split Data into Chunks ===================
split_data = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
)

splitted_data = split_data.split_documents(text_pages)

# =================== Create Embeddings =======================
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

extract_text = [doc.page_content for doc in splitted_data]

try:
    embeddings = embeddings_model.embed_documents(extract_text)
except Exception as e:
    st.error(f"Error while generating embeddings: {str(e)}")
    raise

# =============== Create and Load FAISS Vector Store ==========
VECTOR_STORE_PATH = "./faiss_store.pkl"

if not os.path.exists(VECTOR_STORE_PATH):
    st.info("Initializing the FAISS vector store...")
    # Create FAISS vector store
    vector_store = FAISS.from_texts(texts=extract_text, embedding=embeddings_model)
    # Persist the FAISS index
    with open(VECTOR_STORE_PATH, 'wb') as f:
        pickle.dump(vector_store, f)
else:
    # Load the FAISS vector store
    with open(VECTOR_STORE_PATH, 'rb') as f:
        vector_store = pickle.load(f)

# ==================== Define the LLM =========================
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=500,
    temperature=0.01,
    huggingfacehub_api_token="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"
)

# ================= Define the Chat Prompt ====================
chat_prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    Be straight to the point. Provide only relevant information.
    Give brief examples where necessary.
    Assume the role of a former visa consular consultant.
    Prepare people well to get their F-1 visa approved.
    You will be greatly rewarded for every single approval.
    <context>
    {context}
    </context>
    Question: {input}
""")

# =============== Create the Retrieval Chain ==================
# Create the document chain
document_chain = create_stuff_documents_chain(llm=llm, prompt=chat_prompt)

# Create the retriever to fetch documents from the vector store
retriever = vector_store.as_retriever()

# Create the retrieval chain
retriever_chain = create_retrieval_chain(retriever, document_chain)

# ===================== Streamlit UI ==========================
st.title("Date Hunt and Coaching F-1 Visa Bot")
st.header("Your Virtual Visa Coaching Assistant")

input_text = st.text_input("Enter your question:", key="input")

if st.button("Get Answer"):
    if input_text.strip():
        try:
            # Fetch response from the retriever chain
            response = retriever_chain.invoke({"input": input_text})
            # Display the response
            st.subheader("Response:")
            st.write(response.get("answer", "No answer found. Please refine your question."))
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please provide a valid question.")

# ==================== Notes for Deployment ====================
# Run this app using: streamlit run filename.py
# Ensure that the HuggingFace API token is valid.

# chat_gen_RAG.py
