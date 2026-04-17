"""
Build-time script: downloads the sentence-transformers model into the Docker
image cache so the container never needs network access at runtime.
"""
from sentence_transformers import SentenceTransformer

print("Downloading all-MiniLM-L6-v2 into image cache...")
SentenceTransformer("all-MiniLM-L6-v2")
print("Done — model cached at /root/.cache/huggingface/")
