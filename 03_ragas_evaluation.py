"""
Step 3 — RAGAS Evaluation
=========================
TASK:
  1. Load questions and expected references from golden_set.jsonl
  2. Run them through both prompt versions (V1 and V2)
  3. Build an EvaluationDataset
  4. Evaluate with RAGAS (faithfulness, answer_relevancy, context_recall, context_precision)
  5. Print comparison table and save to data/ragas_report.json

DELIVERABLE: Faithfulness score >= 0.8 for at least one prompt version.
"""

import os
import json
import glob
import sys
import io
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv

# Force UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# ── 1. Setup Models & Prompts ───────────────────────────────────────────────
llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

llm_eval = llm
emb_eval = embeddings

SYSTEM_V1 = (
    "You are a helpful AI assistant for internal company policies. "
    "Answer the user's question using ONLY the provided context. "
    "Keep your answer concise (2-4 sentences). "
    "If the context does not contain the answer, say: 'I don't have enough information.'\n\n"
    "Context:\n{context}"
)
PROMPT_V1 = ChatPromptTemplate.from_messages([("system", SYSTEM_V1), ("human", "{question}")])

SYSTEM_V2 = (
    "You are an expert AI HR/IT tutor for the company. Provide a structured, accurate answer.\n\n"
    "Instructions:\n"
    "1. Read the context carefully.\n"
    "2. Identify the key facts relevant to the question.\n"
    "3. Write a clear, well-organized answer with bullet points if necessary.\n"
    "4. State explicitly if the context lacks sufficient information.\n\n"
    "Context:\n{context}"
)
PROMPT_V2 = ChatPromptTemplate.from_messages([("system", SYSTEM_V2), ("human", "{question}")])


# ── 2. Vectorstore Setup ────────────────────────────────────────────────────
def build_vectorstore():
    txt_files = glob.glob("data/*.txt")
    texts = []
    for file_path in txt_files:
        with open(file_path, "r", encoding="utf-8") as f:
            texts.append(f.read())
            
    full_text = "\n\n".join(texts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(full_text)
    
    return Chroma.from_texts(chunks, embeddings, collection_name="company_knowledge_v3")

def run_chain(retriever, prompt, question):
    docs = retriever.invoke(question)
    context_list = [doc.page_content for doc in docs]
    context_str = "\n\n".join(context_list)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context_str, "question": question})
    return answer, context_list

# ── 3. Main Evaluation Pipeline ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Step 3: RAGAS Evaluation")
    print("=" * 60)

    print("1. Loading golden dataset...")
    qa_data = []
    with open("data/golden_set.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_data.append(json.loads(line))
                
    if not qa_data:
        print("No QA data found!")
        return
        
    print(f"Loaded {len(qa_data)} questions.")

    print("\n2. Building vectorstore...")
    vectorstore = build_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("\n3. Generating answers for both versions...")
    v1_samples = []
    v2_samples = []

    for i, item in enumerate(qa_data):
        q = item["question"]
        ref = item["expected_answer"]
        print(f"   [{i+1}/{len(qa_data)}] Generating answers...")
        
        ans1, ctx1 = run_chain(retriever, PROMPT_V1, q)
        v1_samples.append(SingleTurnSample(user_input=q, response=ans1, retrieved_contexts=ctx1, reference=ref))
        
        ans2, ctx2 = run_chain(retriever, PROMPT_V2, q)
        v2_samples.append(SingleTurnSample(user_input=q, response=ans2, retrieved_contexts=ctx2, reference=ref))

    dataset_v1 = EvaluationDataset(samples=v1_samples)
    dataset_v2 = EvaluationDataset(samples=v2_samples)

    metrics_list = [faithfulness, answer_relevancy, context_recall, context_precision]

    print("\n4. Evaluating V1 with RAGAS...")
    res_v1 = evaluate(dataset_v1, metrics=metrics_list, llm=llm_eval, embeddings=emb_eval)
    
    print("\n5. Evaluating V2 with RAGAS...")
    res_v2 = evaluate(dataset_v2, metrics=metrics_list, llm=llm_eval, embeddings=emb_eval)

    # ── 4. Process and Save Results ─────────────────────────────────────────
    def safe_mean(metric_name, result_obj):
        try:
            scores = result_obj[metric_name.name]
            if isinstance(scores, list):
                clean_scores = [s for s in scores if s is not None and (isinstance(s, (int, float)) and not np.isnan(s))]
                return float(np.mean(clean_scores)) if clean_scores else 0.0
            return float(scores)
        except Exception:
            return 0.0

    scores_v1 = {m.name: safe_mean(m, res_v1) for m in metrics_list}
    scores_v2 = {m.name: safe_mean(m, res_v2) for m in metrics_list}

    print("\n" + "=" * 60)
    print("  RAGAS Evaluation Results")
    print("=" * 60)
    print(f"{'Metric':<20} | {'Prompt V1':<10} | {'Prompt V2':<10}")
    print("-" * 47)
    for m in metrics_list:
        m_name = m.name
        print(f"{m_name:<20} | {scores_v1.get(m_name, 0):<10.4f} | {scores_v2.get(m_name, 0):<10.4f}")
    print("-" * 47)

    if scores_v1.get('faithfulness', 0) >= 0.8 or scores_v2.get('faithfulness', 0) >= 0.8:
        print("\n[SUCCESS] Target met: Faithfulness >= 0.8")

    report = {"v1_scores": scores_v1, "v2_scores": scores_v2}
    with open("data/ragas_report.json", "w") as f:
        json.dump(report, f, indent=4)
        
    print("\nSaved report to data/ragas_report.json")


if __name__ == "__main__":
    main()
