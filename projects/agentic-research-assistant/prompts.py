from langchain.prompts import PromptTemplate

CHUNK_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["chunk"],
    template=(
        "You are an expert research assistant. Summarize the following chunk "
        "into bullet points: problem, methods, results, limitations.\n\n"
        "Chunk:\n{chunk}\n\nSummary:"
    ),
)

REDUCE_SUMMARY_PROMPT = PromptTemplate(
    input_variables=["bullets"],
    template=(
        "Combine these chunk summaries into a cohesive 6–8 bullet executive summary.\n\n"
        "Chunk Summaries:\n{bullets}\n\nFinal Executive Summary:"
    ),
)

QA_PROMPT = PromptTemplate(
    input_variables=["question", "context"],
    template=(
        "Answer the question using ONLY the context. If not present, say 'Not found.'\n\n"
        "Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    ),
)

INSIGHTS_PROMPT = PromptTemplate(
    input_variables=["context"],
    template=(
        "From the context, extract:\n"
        "1) Key contributions\n"
        "2) Methodology overview\n"
        "3) Limitations\n"
        "4) Future work\n\n"
        "Context:\n{context}\n\nInsights:"
    ),
)
