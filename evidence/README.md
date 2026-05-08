# Lab Day 22: RAG Pipeline Observability & Evaluation
**Student Name:** Nguyen Minh Hieu  
**Student ID:** 2A202600180

## Evidence Report

This directory contains the required evidence for the Day 22 Lab, focusing on building a production-grade RAG pipeline using LangChain, ChromaDB, LangSmith, and Guardrails AI.

### 1. Evidence Files Overview

| Filename | Description |
| :--- | :--- |
| `01_langsmith_traces.png` | Screenshot of LangSmith project showing ≥ 50 traces from the initial RAG pipeline. |
| `02_prompt_hub.png` | Screenshot of LangSmith Prompt Hub featuring two prompt versions (V1 & V2). |
| `02_ab_routing_log.txt` | Console log demonstrating deterministic A/B routing across 55 questions. |
| `03_ragas_scores.png` | Screenshot of the RAGAS evaluation comparison table between V1 and V2. |
| `03_ragas_report.json` | Detailed RAGAS metrics in JSON format. |
| `04_pii_demo_log.txt` | Console output showing successful PII detection and redaction. |
| `04_json_demo_log.txt` | Console output showing successful malformed JSON repair. |

### 2. Prompt Analysis: V1 vs. V2

We conducted an A/B test comparing two system prompts:
- **V1 (Concise Assistant):** Focused on short, 2-4 sentence answers.
- **V2 (Structured Expert Tutor):** Focused on structured, well-organized, and expert-level responses.

#### RAGAS Metrics Comparison

| Metric | Prompt V1 (Concise) | Prompt V2 (Structured) |
| :--- | :--- | :--- |
| **Faithfulness** | 0.7545 | **0.8428** |
| **Answer Relevancy** | 0.4003 | **0.4414** |
| **Context Recall** | 0.8796 | **0.8818** |
| **Context Precision** | 0.8040 | **0.8424** |

#### Key Insights

1. **Accuracy & Grounding:** Prompt **V2** achieved a significantly higher **Faithfulness** score (**0.8428**) compared to V1 (0.7545). This indicates that the structured instructions in V2 helped the LLM adhere more strictly to the provided context and reduced hallucinations.
2. **Relevance:** V2 also showed better **Answer Relevancy**, suggesting that structured output (bullet points, clear sections) is more effective at addressing user queries compared to simple concise sentences.
3. **Retrieval Performance:** Both versions showed high Context Recall and Precision (above 0.8), confirming that our **ChromaDB** vector store and retrieval strategy are robust across different prompt styles.

### 3. Conclusion

Based on the RAGAS evaluation, **Prompt V2** is the superior version for production. It successfully met the lab requirement of **Faithfulness ≥ 0.8**, providing more reliable and structured information for internal company policies.
