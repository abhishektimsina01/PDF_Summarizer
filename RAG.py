from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

class LLM_Response(BaseModel):
    answer : str = Field(description="This should containt the answer that we get from the relevant chunks")
    summary : str = Field(description="this should contiaint the summary of the answer that we get in nutshell")

parser_format = PydanticOutputParser(pydantic_object=LLM_Response)

parser = StrOutputParser()

model = ChatGroq(
    model=os.getenv("model"),
    api_key=os.getenv("api_key")
)

file_path = "Documents/Attention.pdf"
loader = PyPDFLoader(file_path=file_path)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 80)
all_split = text_splitter.split_documents(docs)

embeddig_model = OllamaEmbeddings(model="llama3.2")

vector_store = Chroma.from_documents(
    documents=all_split,
    embedding=embeddig_model,
    collection_name="my_collection"
)

retriever = vector_store.as_retriever(search_type = "mmr", search_kwargs = {'k' : 4, "lambda_mult" : 0.7})

template = PromptTemplate(template="""The prompt is {query} and \n 
    {context}, if u dont know the asnwer from the context provided just write i dont know instead of hallucinating and give me short and sweet answer""",
    input_variables=['query', 'context'])

while(True):
    query = input("What is your question? \n")
    if query == "quit":
        break
    contexts = retriever.invoke(query)
    page_content = []
    for i in contexts:
        page_content.append(i.page_content)
    prompt = template.invoke({'query' : query, 'context' : page_content})
    response = model.invoke(prompt)
    print(response.content)