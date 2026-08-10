FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu torch==2.7.1+cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# bake the embedding model into the image so startup does not download it
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
COPY app.py .
EXPOSE 9100
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","9100","--workers","1"]
