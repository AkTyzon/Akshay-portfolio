# Akshay Kaushik

**Machine Learning Engineer — LLM, RAG & agent systems**
📍 Punjab, India · 🎓 M.Tech (Data Science), IIT Jammu · 💼 Scientist B @ C-DOT

🌐 **[aktyzon.github.io/Akshay-portfolio](https://aktyzon.github.io/Akshay-portfolio/)** — portfolio site
📧 2021pcs1016@iitjammu.ac.in · 🔗 [LinkedIn](https://linkedin.com/in/akshay-kaushik-5aa6a8152/)

I build retrieval and agent systems for security operations, and the evaluation tooling to prove
they actually work. Three years turning research-grade ideas into services that stay up.

---

## 💼 Experience

### Scientist B — ML/AI Engineer
**Centre for Development of Telematics (C-DOT)** · Aug 2023 – Present · New Delhi

**Natural-language security analytics platform**
Analysts ask questions in plain English. The system picks the capabilities the question needs, runs
them concurrently across internal security data and five threat-intelligence services, and returns a
grounded answer with its queries and raw rows exposed for verification.

- Deploys against a different organization's data **with no code change** — 40 tables and ~1,700
  described fields resolved at runtime from catalogs, enforced by an automated audit
- Concurrent `plan → execute → synthesize → review` orchestration, so multi-source questions cost the
  slower path rather than the sum
- Self-describing capability registry — adding a capability is one registration instead of three files
  kept manually in sync
- Worst-case query **232s → 66s**, with 62 automated tests written from zero

**Anomaly detection at scale**
- Cut false-positive alerts by **30%** through feature engineering and threshold calibration across
  **5M+ events per endpoint per day**
- Containerized inference services with automated retraining workflows
- Classifiers that catch processes impersonating legitimate system binaries, using allowlists and
  edit-distance matching

**Enterprise knowledge assistant**
- Question answering over **1,000+ internal policy, technical and operational documents**, with source
  citations on every answer so staff can verify rather than trust

### Project Trainee — Machine Learning
**C-DOT** · Jul 2023 – Aug 2023
- Telecom subscriber demand prediction with Random Forest models; feature engineering, validation and
  forecast stability

---

## 🔬 Projects

| Project | What it does | Live |
|---|---|---|
| **[EvalForge](https://136-64-84-253.sslip.io)** | LLM evaluation & benchmarking platform — 18 metrics per answer across classical NLP, retrieval quality and LLM-as-judge scoring | [Demo →](https://136-64-84-253.sslip.io) · [API →](https://136-64-84-253.sslip.io:8443/docs) |
| **[GenAI Résumé Analyzer](https://huggingface.co/spaces/TyzonAk/GEN-AI-RESUME-SUMMARIZER)** | Skill extraction, fit scoring and gap suggestions against a job description, on Llama 3.1 via Groq | [Demo →](https://huggingface.co/spaces/TyzonAk/GEN-AI-RESUME-SUMMARIZER) |
| **[Movie Recommender](./projects/movie-recommender)** | Item-item collaborative filtering on MovieLens — **15.5% hit-rate@10 vs 5.9% popularity baseline (2.63× lift)** | [Demo →](https://aktyzon.github.io/Akshay-portfolio/demos/movie/) |
| **[Recipe Recommender](./projects/food-recommender)** | Content-based TF-IDF over parsed ingredients, tags and nutrition bands — **96% tag agreement@5 vs 38.5% random** | [Demo →](https://aktyzon.github.io/Akshay-portfolio/demos/food/) |
| **[Cascade Learning on Spark](./projects/cascade-learning-spark)** | Two-stage cascade on 581k rows — a cheap model answers what it is confident about, only the residual escalates | [Results →](https://aktyzon.github.io/Akshay-portfolio/demos/cascade/) |

**Masked Face Recognition** — M.Tech thesis. GAN-based framework for face recognition under masked and
occluded conditions; controlled experiments on representation-learning robustness. *(Research work, no
public demo.)*

---

## 🛠 Technical Skills

**ML & GenAI:** LLMs, RAG, agentic systems, prompt engineering, anomaly detection, recommender systems
**Programming:** Python, C++, SQL
**Frameworks:** PyTorch, scikit-learn, TensorFlow, LangChain, LangGraph
**Data & Infra:** Pandas, NumPy, PySpark, Docker, Linux
**Backend:** FastAPI, Flask, Streamlit
**Databases & Search:** PostgreSQL, OpenSearch, Qdrant, FAISS
**Cloud:** GCP (Compute Engine, self-hosted deployments with TLS)

---

## 🎓 Education

**Indian Institute of Technology (IIT), Jammu** — M.Tech, Data Science · CGPA 8.48 · 2023
**Chandigarh University** — B.Tech, Computer Science · CGPA 7.8 · 2020

**Certifications:** IBM — Develop Generative AI Applications · IBM — Agentic AI with LangGraph, CrewAI,
AutoGen & BeeAI
