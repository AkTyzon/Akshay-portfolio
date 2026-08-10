"""
Agentic Research Assistant — arXiv RAG.

Fetches papers from arXiv, chunks and embeds them, retrieves against a question,
and answers with Groq-hosted Llama grounded in the retrieved passages.

Ported from a local Ollama build: generation moved to Groq (no GPU needed) and
embeddings to a CPU sentence-transformer, so it runs on a 1GB free-tier VM.
"""
import os, re, io, time, logging, xml.etree.ElementTree as ET
from typing import List, Dict

import numpy as np
import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("research")

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
CHUNK, OVERLAP, TOP_K = 1200, 150, 4
MAX_PAPERS = 3

app = FastAPI(title="Agentic Research Assistant")
_encoder = None
_cache: Dict[str, dict] = {}


def encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("embedding model loaded")
    return _encoder


def _arxiv_query(search_query: str, k: int) -> List[dict]:
    r = requests.get("http://export.arxiv.org/api/query",
                     params={"search_query": search_query, "start": 0,
                             "max_results": k, "sortBy": "relevance"}, timeout=25)
    r.raise_for_status()
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(r.text).findall("atom:entry", ns):
        link = e.find("atom:id", ns).text.strip()
        out.append({
            "id": link.split("/")[-1],
            "title": re.sub(r"\s+", " ", e.find("atom:title", ns).text).strip(),
            "abstract": re.sub(r"\s+", " ", e.find("atom:summary", ns).text).strip(),
            "url": link,
        })
    return out


def arxiv_search(query: str, k: int = MAX_PAPERS) -> List[dict]:
    """Quote the phrase first.

    Unquoted, arXiv treats "retrieval augmented generation" as loose terms and
    returns anything about augmentation — wearables, cognition — which then
    retrieves badly. Quoting it returns actual papers on the topic. Fall back to
    the loose query only if the phrase matches nothing.
    """
    papers = _arxiv_query(f'all:"{query}"', k)
    if not papers:
        papers = _arxiv_query(f"all:{query}", k)
    return papers


def chunk(text: str) -> List[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + CHUNK])
        i += CHUNK - OVERLAP
    return [c for c in out if len(c.strip()) > 120]


def build_index(papers: List[dict]) -> dict:
    """Abstracts are the index. Full PDFs are far heavier and add little for Q&A."""
    chunks, meta = [], []
    for p in papers:
        for c in chunk(p["abstract"]) or [p["abstract"]]:
            chunks.append(c)
            meta.append({"title": p["title"], "url": p["url"], "id": p["id"]})
    vecs = encoder().encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    return {"chunks": chunks, "meta": meta, "vecs": np.asarray(vecs, dtype=np.float32)}


def retrieve(index: dict, question: str, k: int = TOP_K):
    q = encoder().encode([question], normalize_embeddings=True)[0]
    scores = index["vecs"] @ q                       # cosine, vectors are normalised
    top = np.argsort(-scores)[:k]
    return [{"text": index["chunks"][i], "score": float(scores[i]), **index["meta"][i]} for i in top]


def groq(system: str, user: str, max_tokens: int = 700) -> str:
    if not GROQ_KEY:
        return "GROQ_API_KEY is not configured on this deployment."
    r = requests.post(GROQ_URL, timeout=45,
                      headers={"Authorization": f"Bearer {GROQ_KEY}",
                               "Content-Type": "application/json"},
                      json={"model": MODEL, "temperature": 0.2, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}]})
    if r.status_code != 200:
        log.error("groq %s %s", r.status_code, r.text[:200])
        return f"Upstream model error ({r.status_code})."
    return r.json()["choices"][0]["message"]["content"]


class Ask(BaseModel):
    topic: str
    question: str = ""


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "key_configured": bool(GROQ_KEY)}


@app.post("/api/ask")
def ask(body: Ask):
    t0 = time.time()
    topic = body.topic.strip()
    if not topic:
        return JSONResponse({"error": "topic is required"}, status_code=400)

    key = topic.lower()
    if key not in _cache:
        papers = arxiv_search(topic)
        if not papers:
            return JSONResponse({"error": f"arXiv returned no papers for {topic!r}"}, status_code=404)
        _cache[key] = {"papers": papers, "index": build_index(papers)}
    entry = _cache[key]

    question = body.question.strip() or f"What are the key contributions and open problems in {topic}?"
    # Retrieve on topic + question, not the question alone. A generic question
    # ("what are the open problems?") carries no topical signal, so embedding it
    # by itself matches nothing and every similarity collapses toward zero.
    hits = retrieve(entry["index"], f"{topic}. {question}")
    context = "\n\n".join(f"[{i+1}] {h['title']}\n{h['text']}" for i, h in enumerate(hits))

    answer = groq(
        "You are a research assistant. Answer only from the numbered excerpts provided. "
        "Cite sources inline as [1], [2]. If the excerpts do not contain the answer, say so plainly "
        "instead of guessing.",
        f"Excerpts:\n{context}\n\nQuestion: {question}",
    )
    return {
        "topic": topic, "question": question, "answer": answer,
        "papers": entry["papers"],
        "sources": [{"n": i + 1, "title": h["title"], "url": h["url"], "score": round(h["score"], 3)}
                    for i, h in enumerate(hits)],
        "latency_ms": int((time.time() - t0) * 1000),
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


HTML = """<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Agentic Research Assistant — Akshay Kaushik</title><style>
:root{--bg:#f5f6f3;--surface:#fff;--line:#dde3da;--text:#15231c;--dim:#4a584e;--faint:#7d8b81;
--accent:#0d6b54;--soft:#e7f3ed;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);
font:16.5px/1.62 Charter,Georgia,serif}.w{max-width:840px;margin:0 auto;padding:0 24px}
.bar{border-bottom:1px solid var(--line);background:rgba(245,246,243,.9);position:sticky;top:0}
.bar .w{height:56px;display:flex;align-items:center;justify-content:space-between}
.bar a{font-family:var(--mono);font-size:12.5px;color:var(--dim);text-decoration:none}
h1{font-size:2rem;letter-spacing:-.02em;margin:44px 0 12px}
p.lede{color:var(--dim);margin:0 0 26px;max-width:62ch}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:24px;
box-shadow:0 2px 4px rgba(14,26,20,.04),0 12px 32px rgba(14,26,20,.06)}
label{font-family:var(--mono);font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
color:var(--faint);display:block;margin-bottom:7px}
input{width:100%;padding:12px 14px;border:1px solid var(--line);border-radius:8px;
font:1rem Charter,Georgia,serif;background:var(--bg);margin-bottom:16px}
input:focus{outline:none;border-color:#b7dbca;box-shadow:0 0 0 3px var(--soft)}
button{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:12px 22px;
font-family:var(--mono);font-size:13px;font-weight:600;cursor:pointer}
button:disabled{opacity:.55;cursor:default}
#out{margin-top:26px}.ans{white-space:pre-wrap}
.src{margin-top:20px;padding-top:16px;border-top:1px solid #e8ece5}
.src a{display:block;font-size:.92rem;margin-bottom:9px;color:var(--accent);text-decoration:none}
.src .n{font-family:var(--mono);font-size:11px;color:var(--faint);margin-right:6px}
.meta{font-family:var(--mono);font-size:11.5px;color:var(--faint);margin-top:14px}
.err{color:#a53d3d}footer{padding:36px 0 60px;font-family:var(--mono);font-size:11.5px;color:var(--faint)}
</style></head><body>
<div class=bar><div class=w><a href="https://aktyzon.github.io/Akshay-portfolio/">← Back to portfolio</a>
<span style="font-family:var(--mono);font-size:12px;color:var(--faint)">Llama 3.3 70B · Groq</span></div></div>
<div class=w>
<h1>Agentic Research Assistant</h1>
<p class=lede>Give it a research topic. It searches arXiv, embeds what it finds, retrieves the
passages that actually bear on your question, and answers from those — with citations back to the
papers. If the retrieved text doesn't answer it, it says so rather than inventing something.</p>
<div class=card>
<label for=topic>Research topic</label>
<input id=topic placeholder="retrieval augmented generation" value="retrieval augmented generation">
<label for=q>Question (optional)</label>
<input id=q placeholder="What methods do these papers propose?" value="What methods do these papers propose?">
<button id=go>Search &amp; answer</button>
<div id=out></div>
</div></div>
<footer><div class=w>Akshay Kaushik · arXiv API · sentence-transformers · Groq</div></footer>
<script>
const go=document.getElementById('go'),out=document.getElementById('out');
go.onclick=async()=>{
  const topic=document.getElementById('topic').value.trim();
  if(!topic)return;
  go.disabled=true;out.innerHTML='<p style="color:#7d8b81">Searching arXiv, embedding and retrieving…</p>';
  try{
    const r=await fetch('api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({topic,question:document.getElementById('q').value})});
    const d=await r.json();
    if(d.error){out.innerHTML='<p class=err>'+d.error+'</p>';return;}
    out.innerHTML='<div class=ans>'+d.answer.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))+'</div>'+
      '<div class=src><label>Sources retrieved</label>'+
      d.sources.map(s=>'<a href="'+s.url+'" target=_blank rel=noopener><span class=n>['+s.n+'] sim '+s.score+'</span>'+
        s.title.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]))+'</a>').join('')+
      '</div><div class=meta>'+d.papers.length+' papers indexed · '+d.latency_ms+' ms</div>';
  }catch(e){out.innerHTML='<p class=err>Request failed: '+e.message+'</p>';}
  finally{go.disabled=false;}
};
</script></body></html>"""
