FROM python:3.11-slim

WORKDIR /app

# CPU-only torch: the GPU wheels pull several GB we cannot use here
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# bake the embedding model so first request does not pay a download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY *.py ./

EXPOSE 8501
CMD ["streamlit","run","app.py", \
     "--server.port=8501","--server.address=0.0.0.0", \
     "--server.headless=true","--browser.gatherUsageStats=false", \
     "--server.enableCORS=false","--server.enableXsrfProtection=false"]
