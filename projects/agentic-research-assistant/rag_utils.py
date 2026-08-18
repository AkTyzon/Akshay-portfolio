from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from llm_provider import get_embeddings

# --- Split text into chunks ---
def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

# --- Build FAISS index for one paper ---
def build_faiss(chunks):
    return FAISS.from_texts(chunks, embedding=get_embeddings())

# --- Merge multiple FAISS indexes ---
def merge_faiss(index1, index2):
    """
    Merge two FAISS indexes into one.
    If one is None, return the other.
    """
    if index1 is None:
        return index2
    if index2 is None:
        return index1
    
    index1.merge_from(index2)
    return index1
