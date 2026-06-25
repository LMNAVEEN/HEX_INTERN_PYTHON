#purpose : self correcting rag
import os
import sys
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.2
)

def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250
    )
    return splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(os.path.join(BASE_DIR, "data", "vector_stores", "faiss_index"))

PDF_PATH = os.path.join(BASE_DIR, "data", "pdfs", "Naveen_Devops.pdf")

def initialize_knowledge_base():
    faiss_dir = os.path.join(BASE_DIR, "data", "vector_stores", "faiss_index")
    sentinel = os.path.join(faiss_dir, ".pdf_source")

    if os.path.exists(faiss_dir) and os.path.exists(sentinel):
        with open(sentinel) as f:
            if f.read().strip() == PDF_PATH:
                print("📊 Local index already exists. Skipping build.")
                return
        print("⚠️ Index was built from a different PDF. Rebuilding...")

    print("Parsing PDF and building vector index...")
    raw_text = get_pdf_text(PDF_PATH)
    chunks = get_text_chunks(raw_text)
    get_vector_store(chunks)

    with open(sentinel, "w") as f:
        f.write(PDF_PATH)
    print("✅ FAISS Vector index successfully saved locally.\n")

def load_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    return FAISS.load_local(
        os.path.join(BASE_DIR, "data", "vector_stores", "faiss_index"),
        embeddings,
        allow_dangerous_deserialization=True,
    )

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

def clean_refined_query(text):
    cleaned = text.strip()
    if "Rewritten question:" in cleaned:
        cleaned = cleaned.replace("Rewritten question:", "").strip()
    return cleaned

self_check_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict QA evaluator. Check if the provided context contains direct information "
     "to answer the question. If it does, reply exactly with 'YES'. If it is missing the key details, "
     "reply exactly with 'NO'. Do not add any punctuation or extra words."),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

evaluator_chain = self_check_prompt | llm | StrOutputParser()

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert search query optimizer. Rewrite the user's question into a short list "
     "of 3 to 5 distinct search keywords or phrases optimized for vector database retrieval. "
     "Do not write a conversational sentence. Output ONLY the keywords separated by spaces."),
    ("human", "Original question:\n{question}")
])

rewrite_chain = rewrite_prompt | llm | StrOutputParser() | RunnableLambda(clean_refined_query)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. "
     "Answer the user's question comprehensively using ONLY the provided text context.\n"
     "If the context does not contain the answer, reply exactly with: "
     "'answer is not available in the context'."),
    ("human", "Context:\n{context}\n\nQuestion:\n{input}")
])

generation_chain = answer_prompt | llm | StrOutputParser()

def self_corrective_rag(question):
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    initial_docs = retriever.invoke(question)
    initial_context = format_docs(initial_docs)
    print(f"🔍 Initial Context Retrieved ({len(initial_docs)} chunks).")

    verdict = evaluator_chain.invoke({"context": initial_context, "question": question}).strip().upper()
    print(f"⚖️ Evaluator Verdict: {verdict}")

    def routing_logic(inputs):
        if inputs["verdict"] == "NO":
            print("🔄 Insufficient context. Triggering query rewrite...")
            optimized_query = rewrite_chain.invoke({"question": inputs["original_question"]})
            print(f"🎯 New Query: '{optimized_query}'")
            new_docs = retriever.invoke(optimized_query)
            return {"context": format_docs(new_docs), "input": inputs["original_question"]}
        else:
            print("✅ Initial context sufficient. Proceeding to generation.")
            return {"context": inputs["initial_context"], "input": inputs["original_question"]}

    master_lcel_chain = RunnableLambda(routing_logic) | generation_chain

    answer = master_lcel_chain.invoke({
        "verdict": verdict,
        "initial_context": initial_context,
        "original_question": question
    })

    print(f"\n💬 Answer:\n{answer}")
    return answer

if __name__ == "__main__":
    initialize_knowledge_base()
    q = "What DevOps tools and cloud technologies does Naveen have hands-on experience with, and which projects demonstrate those skills?"
    print(f"\n❓ User Question: {q}")
    print("-" * 60)
    self_corrective_rag(q)
