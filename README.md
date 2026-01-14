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


---



If you want next (later, not now):
- Turn this into **Optimized RAG README**
- Rewrite this in **interview explanation format**
- Add **diagram-only version**

For now — this README is **clean, honest, and GitHub-safe** ✅
