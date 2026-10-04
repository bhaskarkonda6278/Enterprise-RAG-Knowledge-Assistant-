from pathlib import Path
from typing import List
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq


class RAGEngine:
    """Lightweight RAG engine:
    documents -> chunks -> TF-IDF vectors -> similarity retrieval -> LLM
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.chunks: List[str] = []
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
        self.matrix = None
        self._load_documents()

    def _load_documents(self):
        documents = []

        for path in sorted(self.data_dir.glob("*.txt")):
            documents.append(path.read_text(encoding="utf-8"))

        self.chunks = self._chunk_documents(documents)

        if self.chunks:
            self.matrix = self.vectorizer.fit_transform(self.chunks)

    @staticmethod
    def _chunk_documents(documents, chunk_size=120, overlap=20):
        chunks = []

        for document in documents:
            words = document.split()

            for start in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(
                    words[start:start + chunk_size]
                )

                if chunk.strip():
                    chunks.append(chunk)

        return chunks

    def retrieve(self, query: str, top_k: int = 3):
        if not self.chunks:
            return []

        query_vector = self.vectorizer.transform([query])

        scores = (self.matrix @ query_vector.T).toarray().ravel()

        indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "text": self.chunks[i],
                "score": float(scores[i])
            }
            for i in indices
        ]

    def generate(self, query: str, top_k: int = 3):
        retrieved = self.retrieve(query, top_k)

        if not retrieved:
            return {
                "answer": "No documents are available.",
                "sources": []
            }

        context = "\n\n---\n\n".join(
            item["text"] for item in retrieved
        )

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            return {
                "answer": "Retrieval succeeded, but GROQ_API_KEY is not configured.",
                "sources": retrieved
            }

        client = Groq(api_key=api_key)

        prompt = f"""
Answer the user's question using ONLY the supplied context.

If the context does not contain enough information,
say that the information is not available in the supplied documents.

Context:
{context}

Question:
{query}
"""

        response = client.chat.completions.create(
            model=os.getenv(
                "GROQ_MODEL",
                "llama-3.1-8b-instant"
            ),
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise enterprise knowledge assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": retrieved
        }