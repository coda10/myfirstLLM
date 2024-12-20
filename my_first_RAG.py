from langchain_community.document_loaders import TextLoader
# from langchain_chroma import Chroma
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.embeddings import HuggingFaceInferenceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

# Load Data
load_text=TextLoader('./textfile.txt', encoding='utf-8')
text_loader = load_text.lazy_load()
text_pages = []

for page in text_loader:
    text_pages.append(page)
    

# Split Data into chuncks

split_data = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
)

splitted_data = split_data.split_documents(text_pages)

# Create embeddings
# embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

embeddings_model = HuggingFaceInferenceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


extract_text = [doc.page_content for doc in splitted_data]
embeddings = embeddings_model.embed_documents(extract_text)

# =============================================================================
# Create the persisted vector store
# vector_store = Chroma.from_texts(
#     texts=extract_text,  # The list of texts (documents)
#     embedding=embeddings_model,  # The embeddings function
#     persist_directory="./chroma_store"  # Optional: Directory for persistence
# )

# # Load the persisted vector store
# vector_store = Chroma(
#     persist_directory="./chroma_store",
#     embedding_function=embeddings_model
# )

vector_store = FAISS.from_texts(texts=extract_text, embedding=embeddings_model)

# =========================================================================
# Define the template
chat_prompt = ChatPromptTemplate.from_template("""
                Answer the following question based only on the provided context.
                Be straight to the point. Give only revelant information.
                Give brief examples where necessary.
                Assume the role of a former visa consular consultant.
                Prepare people well to get their F-1 visa approved.
                You will be greatly rewarded for every single approval.
                <context>
                {context}
                </context>
                Question: {input}
            """)

llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=500,
    temperature=0.01,
    huggingfacehub_api_token="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" 
)

# =========================================================================

# Create CHAIN
document_chain = create_stuff_documents_chain(llm=llm, prompt=chat_prompt)

# Create Retriever to retrieve data from db
retriever = vector_store.as_retriever()

# Create Retriever Chain that wraps document_chain/CHAIN and retriever
# Create Retriever pass user input to retriever, receives response/Documents and pass them to LLM

retriever_chain = create_retrieval_chain(retriever, document_chain)

# query = "why are you going to the United States?"

# response = retriever_chain.invoke({ "input": query})

# print(response["answer"])

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
    

# streamlit run my_first_RAG.py

# pip freeze > requirements.txt

# pip install -r requirements.txt