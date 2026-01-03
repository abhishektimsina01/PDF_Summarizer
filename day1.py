from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

file_path = "Attention.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()
print("Document", len(docs))
for i in range(len(docs)):
    print("--------------------------------------------")
    print(docs[i])

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 80, add_start_index = True)
all_split = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="llama3.2")

print("Chunks", len(all_split))
print(len(all_split[55].page_content)) 
print(all_split[55].page_content)

print("-----------------------------------------------------------------------------------------------------")

embeddings = OllamaEmbeddings(model="llama3.2")

vector_store = InMemoryVectorStore(embeddings)

ids = vector_store.add_documents(documents=all_split)
# print(len(ids))
query = input("What do u wanna ask:   ")
results = vector_store.similarity_search(query, k = 4)
print("-----------------------------------------------------------------------------------------------------")
print("the top 3 chunks are")
page_content = []
for i in results:
    print(i.page_content)
    print("-----------------------------------------------------------------------------------------------------")
    page_content.append(i.page_content)

template = PromptTemplate(
    input_variables=["query", "page_content"],
    template = """Assume yourself as an expert in polices and law. 
            Use the context below to answer the question. Only use information from the context, and do not make up any information. 
            Provide a clear and concise answer in 7-8 sentences. If the answer is not in the context, reply 'Information not found'.
            Don't hallucinate. Give me answer in nepali
            
            Context (in array): {page_content}
            Question: {query} """
)
prompt = template.format(query = query, page_content = page_content)

llm = ChatGroq(
    model="groq/compound-mini", 
    temperature=0,  
    api_key=os.getenv("api_key")
)
response = llm.invoke(prompt)
print("-----------------------------------------------------------------------------------------------------")
print(response.content)  