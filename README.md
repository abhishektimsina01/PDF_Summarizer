# 📘 Traditional Retrieval-Augmented Generation (RAG)

This repository contains a **simple Traditional RAG implementation using LangChain**.  
The goal of this project is to understand **how RAG actually works under the hood**, why each step exists, and how different design choices (chunking, embeddings, vector stores, retrievers) affect the final answer quality.

No agents, no complex loops — just a clean, foundational RAG pipeline.

---

## ❓ What is RAG?

RAG (Retrieval-Augmented Generation) is a technique where an LLM does **not rely only on its training data**, but instead:

1. Retrieves relevant information from external documents
2. Uses that retrieved context to generate a more accurate answer

---

## 🤔 Why Do We Need RAG?

LLMs alone have limitations:
- They don’t know your private data
- They can hallucinate
- Their knowledge can be outdated

RAG solves this by:
- Grounding responses in real documents
- Reducing hallucinations
- Making LLMs usable for real applications

---

## 🏗️ Traditional RAG Pipeline (As Used in This Project)

Documents
→ Chunking
→ Embeddings
→ Vector Store
→ Retriever
→ LLM

---



Below is **what happens at each step and why it exists**.

---

## ✂️ Chunking

### Why Chunking Is Needed

Documents are usually too large to:
- Embed as a whole
- Fit inside an LLM context window

Chunking breaks documents into **smaller, meaningful pieces** so they can be:
- Embedded properly
- Retrieved accurately

---

### Types of Chunking

#### 1. Fixed / Character Chunking
Splits text based on fixed length.

- Simple
- Can break meaning in the middle of sentences

---

#### 2. Token-Based Chunking
Splits based on token count.

- LLM-aware
- Still ignores document structure

---

#### 3. Recursive Chunking ✅ (Used)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)
```

---


Why I Chose Recursive Chunking

Preserves semantic structure

Tries paragraphs → sentences → words

Reduces broken context

Works well for real documents (PDFs, notes, articles)

For traditional RAG, this gives the best retrieval quality with minimal complexity.

🧠 Embeddings

Embeddings convert text into vectors so we can compare meaning mathematically.

When Is an Embedding Model “Good”?

A good embedding model:

Places semantically similar text close together

Separates unrelated concepts clearly

Is consistent across different phrasings

About Embedding Dimensions

Higher dimensions ≠ always better

More dimensions = more expressive, but:

More memory

Slower search

Lower dimensions:

Faster

Less precise

📌 In practice:

Small projects → lower dimensions are fine

Large, complex knowledge bases → higher dimensions help

Balance quality vs performance, not just size.

🗄️ Vector Stores

Vector stores store embeddings and allow similarity search.

1. FAISS (Local, In-Memory)

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(chunks, embeddings)
```


Use when:

Learning RAG

Local experiments

Fast prototyping

2. Chroma (Persistent Local Storage)
```python
from langchain.vectorstores import Chroma


vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)
```


Use when:

You want persistence

Small to medium projects

Simple setup without cloud services

3. Pinecone (Managed, Cloud)
```python
from langchain.vectorstores import Pinecone

vectorstore = Pinecone.from_documents(
    chunks,
    embeddings,
    index_name="rag-index"
)
```

Use when:

Large-scale data

Production systems

Distributed access

🔎 Retrievers

Retrievers decide which chunks are shown to the LLM.
This step has the biggest impact on answer quality.

1. Single-Query Similarity Search
```python
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```


How it works

Embeds the user query

Fetches top-k closest chunks

Limitations

Can miss relevant context

Often retrieves repetitive chunks

2. MMR (Max Marginal Relevance)
```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 15
    }
)
```

Why MMR Is Better Than Simple Similarity

Balances relevance + diversity

Avoids near-duplicate chunks

Covers more aspects of the query

3. Multi-Query Retrieval (Best)
```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(search_type="mmr"),
    llm=llm
)
```

Why Multi-Query Is Better Than MMR Alone

LLM rewrites the query in multiple ways

Retrieves context from different perspectives

Fixes embedding blind spots

📌 MMR improves quality, Multi-Query improves coverage
Together, they significantly improve traditional RAG results.

🔧 How This Can Be Improved (Agentic RAG Idea)

Traditional RAG is static:

One retrieval pass

No reasoning about whether results are good

No memory

Agentic RAG improves this by:

Letting an agent decide when to retrieve

Re-retrieving if context is weak

Using tools dynamically

Adding memory across steps

This project focuses on strong foundations, which is required before moving to agentic systems.

📌 Summary

This repository focuses on:

Understanding why each RAG step exists

Making correct design choices

Building a clean Traditional RAG using LangChain

This is the base layer for:

Optimized RAG

Agentic RAG

Production LLM systems
