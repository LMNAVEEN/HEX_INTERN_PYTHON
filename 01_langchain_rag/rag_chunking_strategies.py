# Run: python 01_langchain_rag/rag_chunking_strategies.py
# Tech: LangChain + FAISS + HuggingFace Embeddings + Groq (LLaMA)
# Purpose: Compares Recursive, Character, and Spacy chunking strategies on a PDF using RAG

import sys
import os
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SpacyTextSplitter, CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
PDF_PATH    = os.path.join(BASE_DIR, "data", "pdfs", "product_manual.pdf")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL   = "llama-3.3-70b-versatile"
QUESTION    = "What is the main purpose of this product?"
TOP_K       = 5
# ─────────────────────────────────────────────────────────────────────────────


#Embedding statergy
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    show_progress=True,
    encode_kwargs={"batch_size": 64},
)

#llm model
llm = ChatGroq(model=LLM_MODEL, temperature=1)


#chat prompt template
chain = ChatPromptTemplate.from_messages([
    ("system", "You are a precise assistant. Answer strictly using the provided context. "
               "If the answer is not present, say 'Not found in document.'"),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
]) | llm | StrOutputParser()


# FAISS reteiver for fetching or creating if not available
def build_retriever(name, splitter, docs):
    cache = os.path.join(BASE_DIR, "data", "vector_stores", f"faiss_cache_{name.lower()}")
    if os.path.exists(cache):
        store = FAISS.load_local(cache, embeddings, allow_dangerous_deserialization=True)
    else:
        chunks = splitter.split_documents(docs)
        store  = FAISS.from_documents(chunks, embeddings)
        store.save_local(cache)
    return store.as_retriever(search_kwargs={"k": TOP_K})

# main function to print the outputs of 3 chunking statergies
def main():
    docs = PyMuPDFLoader(PDF_PATH).load()
    print(f"Loaded {len(docs)} pages from '{PDF_PATH}'")
    print(f"Question: {QUESTION}\n")

# 3 types of chunking statergies
    splitters = {
        "Recursive": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
        "Character": CharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
        "Spacy":     SpacyTextSplitter(chunk_size=1000),
    }

    for name, splitter in splitters.items():
        retriever = build_retriever(name, splitter, docs)
        n_chunks  = retriever.vectorstore.index.ntotal
        context   = "\n\n".join(d.page_content for d in retriever.invoke(QUESTION))
        answer    = chain.invoke({"context": context, "question": QUESTION})
        print(f"{name} ({n_chunks} chunks)")
        print(answer)
        print()


if __name__ == "__main__":
    main()