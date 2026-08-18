import streamlit as st
from llm_provider import provider_name
from pdf_utils import extract_text_from_pdf
from rag_utils import chunk_text, build_faiss, merge_faiss
from agent import map_reduce_summary, rag_answer, rag_insights
from fetch_papers import fetch_arxiv_pdf, fetch_papers_by_keyword

# ------------------ Streamlit Config ------------------
st.set_page_config(page_title="Agentic Research Assistant", page_icon="🧠", layout="wide")
st.title("🧠 Agentic Research Assistant")
st.caption(f"Multi-paper RAG over full paper text · {provider_name()}")

# ------------------ Session State ------------------
for key in ["chunks", "vs", "summaries", "answer", "sources", "insights", "results", "selected_papers"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["summaries", "selected_papers"] else None

# ------------------ Upload or Fetch ------------------
st.subheader("📚 Upload or Fetch Papers")

uploaded_files = st.file_uploader("Upload research PDFs", type=["pdf"], accept_multiple_files=True)
arxiv_ids = st.text_area("Or enter multiple arXiv IDs (comma-separated, e.g., 2307.03172, 2205.12345)")
keyword = st.text_input("Or search arXiv by keyword (e.g., LLM RAG)")
num_results = st.slider("How many recent papers?", 1, 5, 3)

# Keyword search
if keyword.strip() and st.button("🔍 Search Papers"):
    with st.spinner("Searching arXiv..."):
        results = fetch_papers_by_keyword(keyword.strip(), max_results=num_results)
        if results:
            st.session_state.results = results
            st.session_state.selected_papers = [r["id"] for r in results]  # default select all
        else:
            st.error("No papers found for this keyword.")

# Display search results with checkboxes
if st.session_state.results:
    st.markdown("### 📑 Select Papers from Search Results")
    selected = []
    for r in st.session_state.results:
        if st.checkbox(f"{r['title']} ({r['id']})", value=True):
            selected.append(r["id"])
    st.session_state.selected_papers = selected

# ------------------ Build Knowledge Base ------------------
if st.button("🔧 Build Knowledge Base"):
    with st.spinner("Building KB from selected papers..."):
        all_chunks = []
        all_vs = None
        st.session_state.summaries = []  # reset

        # PDFs
        if uploaded_files:
            for file in uploaded_files:
                text = extract_text_from_pdf(file)
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                vs = build_faiss(chunks)
                all_vs = merge_faiss(all_vs, vs)

                # Per-paper summary
                summary = map_reduce_summary(chunks)
                st.session_state.summaries.append(summary)

        # arXiv IDs
        if arxiv_ids.strip():
            for aid in [a.strip() for a in arxiv_ids.split(",")]:
                text = fetch_arxiv_pdf(aid)
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                vs = build_faiss(chunks)
                all_vs = merge_faiss(all_vs, vs)

                summary = map_reduce_summary(chunks)
                st.session_state.summaries.append(summary)

        # Keyword search selections
        if st.session_state.selected_papers:
            for aid in st.session_state.selected_papers:
                text = fetch_arxiv_pdf(aid)
                chunks = chunk_text(text)
                all_chunks.extend(chunks)
                vs = build_faiss(chunks)
                all_vs = merge_faiss(all_vs, vs)

                summary = map_reduce_summary(chunks)
                st.session_state.summaries.append(summary)

        if not all_chunks:
            st.error("Please upload PDFs or select at least one paper.")
            st.stop()

        st.session_state.chunks = all_chunks
        st.session_state.vs = all_vs
    st.success("✅ Knowledge Base ready!")

# ------------------ Insights ------------------
if st.button("🔎 Extract Cross-Paper Insights") and st.session_state.vs:
    with st.spinner("Extracting insights across all papers..."):
        st.session_state.insights, _ = rag_insights(st.session_state.vs)

# ------------------ QA ------------------
question = st.text_input("💬 Ask a question across ALL papers")
if st.button("❓ Ask") and st.session_state.vs and question.strip():
    with st.spinner("Retrieving answer from KB..."):
        st.session_state.answer, st.session_state.sources = rag_answer(st.session_state.vs, question)

# ------------------ Display Results ------------------
if st.session_state.summaries:
    st.subheader("📝 Executive Summaries")
    for i, summary in enumerate(st.session_state.summaries, 1):
        st.markdown(f"**Summary {i}:** {summary}")

if st.session_state.insights:
    st.subheader("🔎 Cross-Paper Insights")
    st.write(st.session_state.insights)

if st.session_state.answer:
    st.subheader("✅ Answer")
    st.write(st.session_state.answer)
    with st.expander("📚 Sources"):
        for i, src in enumerate(st.session_state.sources, 1):
            st.markdown(f"**Source {i}:** {src[:300]}...")
