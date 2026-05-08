"""
Step 1 — LangSmith-instrumented RAG Pipeline
=============================================
TASK:
  1. Load your dataset, split into chunks, index with ChromaDB
  2. Build a RAG chain: retriever → prompt → LLM → output parser
  3. Decorate the query function with @traceable so every call is traced
  4. Run all questions from golden_set.jsonl → generates traces

DELIVERABLE: Open https://smith.langchain.com and confirm traces appear.
"""

import os
import json
import glob
import sys
import io
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── 1. Environment setup ────────────────────────────────────────────────────
load_dotenv()

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"]  = "true"
os.environ["LANGCHAIN_API_KEY"]     = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]     = os.getenv("LANGCHAIN_PROJECT", "day22-lab-vn")
os.environ["LANGCHAIN_ENDPOINT"]    = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

# ── 2. LangChain + LangSmith imports ────────────────────────────────────────
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable

# ── 3. LLM and Embeddings ───────────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


# ── 4. Build ChromaDB vector store ──────────────────────────────────────────
def build_vectorstore():
    """
    Load the knowledge base from multiple .txt files, split into chunks, embed and index with ChromaDB.
    """
    print("Loading data from data/*.txt...")
    txt_files = glob.glob("data/*.txt")
    texts = []
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
            
    full_text = "\n\n".join(texts)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)
    print(f"Split into {len(chunks)} chunks")

    # Use ChromaDB
    vectorstore = Chroma.from_texts(chunks, embeddings, collection_name="company_knowledge")
    return vectorstore


# ── 5. RAG prompt template ──────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the context below to answer.\n\nContext:\n{context}"),
    ("human",  "{question}"),
])


# ── 6. Build the RAG chain ──────────────────────────────────────────────────
def build_rag_chain(vectorstore):
    """
    Build a LangChain RAG chain using LCEL (pipe operator).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── 7. Traced query function ────────────────────────────────────────────────
@traceable(name="rag-query", tags=["rag", "step1"])
def ask(chain, question: str) -> str:
    """
    Run the RAG chain on a single question.
    The @traceable decorator sends input/output/latency to LangSmith.
    """
    return chain.invoke(question)


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 1: LangSmith RAG Pipeline")
    print("=" * 60)

    # 1. Build the vectorstore
    vectorstore = build_vectorstore()

    # 2. Build the RAG chain
    chain, retriever = build_rag_chain(vectorstore)

    # 3. Load questions from golden_set.jsonl
    sample_questions = []
    golden_set_path = "data/golden_set.jsonl"
    if os.path.exists(golden_set_path):
        with open(golden_set_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sample_questions.append(item["question"])
    else:
        print(f"Warning: {golden_set_path} not found.")

    if not sample_questions:
        print("No questions found to run.")
        return

    # 4. Loop through questions, call ask(), print results
    for i, question in enumerate(sample_questions, 1):
        answer = ask(chain, question)
        print(f"[{i:02d}/{len(sample_questions)}] Q: {question[:60]}")
        print(f"       A: {answer[:100]}...\n")

    # 5. Print confirmation that traces were sent
    print(f"[SUCCESS] {len(sample_questions)} traces sent to LangSmith project '{os.environ['LANGCHAIN_PROJECT']}'")
    print("   Open https://smith.langchain.com to view traces.")


if __name__ == "__main__":
    main()
