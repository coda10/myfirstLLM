from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_huggingface import HuggingFaceEndpoint
import streamlit as st

# Define the template
template = """
You are a helpful assistant. Answer the following question but being straight forward to the point.
Any question out side Technology response by saying that subject matter is out of scope and suggest 
information about that subject matter can be found. Don't allow "".
Question: {question}
"""

# Create a PromptTemplate
prompt = PromptTemplate(
    input_variables=["question"],  # Variables to pass into the template
    template=template,            # Template structure
)

# Format the prompt
# formatted_prompt = prompt.format(question="What is Deep Learning?")
# print(formatted_prompt)

llm = HuggingFaceEndpoint(
    # endpoint_url="https://api-inference.huggingface.co/models/openai-community/gpt2",
    # endpoint_url="https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B",
    endpoint_url="https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct",
    max_new_tokens=2000,
    temperature=0.01,
    huggingfacehub_api_token="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" 
)
# query= "is this code correct: car.[color] === red"
# query= "what is the capital city of Ghana, be straight to the point"

# response = llm.invoke(query)

# print(response)
result_chain = LLMChain(llm=llm, prompt=prompt)

st.set_page_config(page_title="First Bot")
st.header("Daniel's Chat Bot")

## Input
input = st.text_input("Input: ", key = "input")

final_result = result_chain.invoke(input)

## Response
submit=st.button("Answer")
if submit:
    st.subheader("The Response is: ")
    st.write(final_result["text"])

# query= "what is the capital city of Ghana"
# query= "is laptop a computer?"

# final_result = result_chain.invoke(query)

# print(final_result)

# streamlit run main.py