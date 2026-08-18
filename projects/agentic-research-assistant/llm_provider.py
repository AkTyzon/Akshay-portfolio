"""
LLM and embedding providers.

The app was written against a local Ollama server. That still works — but it
cannot run on a machine without Ollama installed, which rules out any free
hosting. So provider selection is now driven by environment:

    GROQ_API_KEY set  ->  Groq (hosted, no GPU needed)  + CPU sentence-transformer embeddings
    otherwise         ->  Ollama (local, as originally written)

Nothing else in the codebase needs to know which one is active.
"""
import os
import logging

log = logging.getLogger(__name__)

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mistral")
OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

# Hosted model names get retired without warning — pinning one means the app
# dies with a 404 the day it is removed, and the failure looks like a bug in
# our code rather than a change upstream. So resolve against the live catalogue
# at startup and fall back down this list.
GROQ_PREFERENCE = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
    "groq/compound",
]

_llm = None
_embeddings = None
_resolved_model = None


def using_groq() -> bool:
    return bool(GROQ_KEY)


def resolve_groq_model() -> str:
    """Pick the first preferred model the account can actually serve."""
    global _resolved_model
    if _resolved_model:
        return _resolved_model

    override = os.environ.get("GROQ_MODEL")
    if override:
        _resolved_model = override
        return _resolved_model

    try:
        import requests
        r = requests.get("https://api.groq.com/openai/v1/models",
                         headers={"Authorization": f"Bearer {GROQ_KEY}"}, timeout=15)
        r.raise_for_status()
        available = {m["id"] for m in r.json().get("data", [])}
        for name in GROQ_PREFERENCE:
            if name in available:
                _resolved_model = name
                log.info("Groq model resolved to %s", name)
                return name
        log.warning("None of the preferred models are available; account offers: %s",
                    sorted(available))
    except Exception as e:                      # network down, bad key, API change
        log.warning("Could not query the Groq model list (%s); using first preference", e)

    _resolved_model = GROQ_PREFERENCE[0]
    return _resolved_model


def get_llm():
    """Chat model. Groq when a key is present, else local Ollama."""
    global _llm
    if _llm is not None:
        return _llm

    if using_groq():
        from langchain_groq import ChatGroq
        model = resolve_groq_model()
        # these models emit reasoning tokens before the answer, so a small cap
        # returns an empty string rather than an error
        _llm = ChatGroq(model=model, api_key=GROQ_KEY, temperature=0.2, max_tokens=1024)
        log.info("LLM: Groq %s", model)
    else:
        from langchain_ollama import OllamaLLM
        _llm = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
        log.info("LLM: Ollama %s at %s", OLLAMA_MODEL, OLLAMA_URL)
    return _llm


def get_embeddings():
    """Embeddings. Groq serves no embedding endpoint, so hosted mode uses a
    CPU sentence-transformer — small enough to run inside a 1 GB VM."""
    global _embeddings
    if _embeddings is not None:
        return _embeddings

    if using_groq():
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        log.info("Embeddings: sentence-transformers %s", EMBED_MODEL)
    else:
        from langchain_community.embeddings import OllamaEmbeddings
        _embeddings = OllamaEmbeddings(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
        log.info("Embeddings: Ollama %s", OLLAMA_MODEL)
    return _embeddings


def provider_name() -> str:
    return f"Groq · {resolve_groq_model()}" if using_groq() else f"Ollama · {OLLAMA_MODEL} (local)"
