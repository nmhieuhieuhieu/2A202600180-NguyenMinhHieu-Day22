"""
Step 2 — Prompt Hub & A/B Routing
===================================
TASK:
  1. Write two distinct system prompts (V1: concise, V2: structured)
  2. Push both to LangSmith Prompt Hub via client.push_prompt()
  3. Pull them back via client.pull_prompt()
  4. Implement deterministic A/B routing: hash(request_id) % 2 → V1 or V2
  5. Run all questions through the router → ≥ 50 more LangSmith traces

DELIVERABLE: 2 named prompts visible in https://smith.langchain.com Prompt Hub
"""

import os
import sys
import io
import json
import hashlib
import glob
from pathlib import Path
from dotenv import load_dotenv

# Force UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── 1. Environment / imports ────────────────────────────────────────────────
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "day22-lab-vn")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import Client, traceable

# ── 2. Define two prompt templates ──────────────────────────────────────────
SYSTEM_V1 = (
    "You are a helpful AI assistant for internal company policies. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "You are an expert AI HR/IT tutor for the company. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer with bullet points if necessary.\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])

PROMPT_V1_NAME = "company-rag-prompt-v1"
PROMPT_V2_NAME = "company-rag-prompt-v2"


# ── 3. Push prompts to LangSmith Prompt Hub ──────────────────────────────────
def push_prompts_to_hub(client):
    """Upload both prompt versions to LangSmith Prompt Hub."""
    try:
        url1 = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – concise internal policy answers")
        print(f"[SUCCESS] Pushed V1 -> {url1}")
    except Exception as e:
        print(f"[ERROR] V1: {e}")

    try:
        url2 = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – structured internal policy answers")
        print(f"[SUCCESS] Pushed V2 -> {url2}")
    except Exception as e:
        print(f"[ERROR] V2: {e}")


# ── 4. Pull prompts from Prompt Hub ─────────────────────────────────────────
def pull_prompts_from_hub(client):
    """Download both prompt versions from LangSmith Prompt Hub."""
    prompts = {}

    try:
        prompts[PROMPT_V1_NAME] = client.pull_prompt(PROMPT_V1_NAME)
        print(f"[INFO] Pulled '{PROMPT_V1_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V1_NAME] = PROMPT_V1
        print(f"[INFO] Using local fallback for '{PROMPT_V1_NAME}'")

    try:
        prompts[PROMPT_V2_NAME] = client.pull_prompt(PROMPT_V2_NAME)
        print(f"[INFO] Pulled '{PROMPT_V2_NAME}' from Hub")
    except Exception:
        prompts[PROMPT_V2_NAME] = PROMPT_V2
        print(f"[INFO] Using local fallback for '{PROMPT_V2_NAME}'")

    return prompts


# ── 5. A/B routing — deterministic hash ─────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Route a request to prompt V1 or V2 based on the MD5 hash of request_id.
    even hash → PROMPT_V1_NAME
    odd  hash → PROMPT_V2_NAME
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Build vectorstore (reuse from step 1) ────────────────────────────────
def build_vectorstore():
    embeddings = OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )
    
    txt_files = glob.glob("data/*.txt")
    texts = []
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
            
    full_text = "\n\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)
    
    vectorstore = Chroma.from_texts(chunks, embeddings, collection_name="company_knowledge_v2")
    return vectorstore


# ── 7. Traced A/B query function ────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """Run the RAG chain using the given prompt version."""
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})

    return {"question": question, "answer": answer, "version": version}


# ── 8. Main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    client = Client(api_key=os.environ["LANGCHAIN_API_KEY"])

    print("1. Pushing prompts...")
    push_prompts_to_hub(client)

    print("\n2. Pulling prompts...")
    prompts = pull_prompts_from_hub(client)

    print("\n3. Building Vectorstore...")
    vectorstore = build_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
    )

    print("\n4. Running A/B Tests...")
    sample_questions = []
    golden_set_path = "data/golden_set.jsonl"
    if os.path.exists(golden_set_path):
        with open(golden_set_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sample_questions.append(item["question"])

    v1_count = 0
    v2_count = 0

    for i, question in enumerate(sample_questions):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1

        result = ask_ab(retriever, llm, prompt, question, version_tag)
        print(f"[{i+1:02d}] [prompt-{version_tag}] Q: {question[:55]}...")
        print(f"       A: {result['answer'][:80]}...\n")

    print("=" * 60)
    print(f"Routing Summary: V1={v1_count}, V2={v2_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
