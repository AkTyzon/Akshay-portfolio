from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.schema import Document

from llm_provider import get_llm

# Provider is chosen by environment: Groq when GROQ_API_KEY is set, else local Ollama.
llm = get_llm()


def _text(response):
    """Normalise a model response to a plain string.

    Ollama is an LLM and returns str; Groq is a chat model and returns an
    AIMessage. Callers (and Streamlit) want text either way.
    """
    return getattr(response, "content", response)
# --- Summarization ---
def map_reduce_summary(chunks, paper_title="Research Paper"):
    """Summarize a single paper using map-reduce."""
    docs = [Document(page_content=c) for c in chunks]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    summary = chain.run(docs)
    return f"### 📄 {paper_title}\n{summary}"

def summarize_multiple(papers):
    """
    Summarize multiple papers.
    papers = [(title, chunks), ...]
    """
    summaries = []
    for title, chunks in papers:
        summaries.append(map_reduce_summary(chunks, paper_title=title))
    return "\n\n".join(summaries)

# --- RAG QA ---
def rag_answer(vectorstore, question):
    """
    Ask a question across ALL papers in the vectorstore.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are a helpful research assistant. 
    Use the following context from multiple research papers to answer the question.

    Context:
    {context}

    Question: {question}
    Answer in a concise, well-structured way:
    """
    response = llm.invoke(prompt)
    return _text(response), [d.page_content for d in docs]

# --- RAG Insights ---
def rag_insights(vectorstore):
    """
    Extract cross-paper insights (themes, trends).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    docs = retriever.get_relevant_documents("summarize key insights and contributions")
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
    You are an expert AI researcher. 
    From the following research paper excerpts, extract **3-5 key insights** 
    (e.g., methods, results, open problems, applications).

    Context:
    {context}

    Write in bullet points.
    """
    response = llm.invoke(prompt)
    return _text(response), [d.page_content for d in docs]
