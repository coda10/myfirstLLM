# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader

# Load Data
load_text=TextLoader('./textfile.txt', encoding='utf-8')
text_loader = load_text.lazy_load()
text_pages = []

for page in text_loader:
    text_pages.append(page)
    
# print(text_pages[0])

# Split Data into chuncks
from langchain.text_splitter import RecursiveCharacterTextSplitter
split_data = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=150,
    # length_function=len,
    # is_separator_regex=False,
)

splitted_data = split_data.split_documents(text_pages)
# print(splitted_data[:3])

# Create embeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

# endpoint_url = "https://huggingface.co/sentence-transformers/all-mpnet-base-v2"
# embeddings_model = HuggingFaceEndpointEmbeddings(endpoint_url=endpoint_url)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

extract_text = [doc.page_content for doc in splitted_data]
embeddings = embeddings_model.embed_documents(extract_text)
# print(len(embeddings))

# Create Vector Store
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

# =============================================================================
# Create the persisted vector store
# vector_store = Chroma.from_texts(
#     texts=extract_text,  # The list of texts (documents)
#     embedding=embeddings_model,  # The embeddings function
#     persist_directory="./chroma_store"  # Optional: Directory for persistence
# )

# =============================================================================
# Load the persisted vector store
vector_store = Chroma(
    persist_directory="./chroma_store",
    embedding_function=embeddings_model
)

# print(vector_store)

# Save in Vector Store

# Do Similarity Check
query = "why are you going to the United States?"
docs = vector_store.similarity_search(query=query)

# print(docs)

# streamlit run main.py

# pip freeze > requirements.txt

# pip install -r requirements.txt