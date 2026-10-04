# Enterprise-RAG-Knowledge-Assistant-
# Enterprise RAG Knowledge Assistant

An end-to-end Retrieval-Augmented Generation (RAG) application for answering questions from enterprise-style technical documentation.

The system combines document processing, TF-IDF text representation, similarity-based retrieval, prompt grounding, and an LLM to generate answers based on retrieved source content. A FastAPI service exposes the RAG pipeline through a validated REST API, with automated testing and evaluation workflows.

## Project Overview

Enterprise organisations often store large amounts of technical information in documents, reports, manuals, and internal knowledge bases. Finding the relevant information manually can be slow and inconsistent.

This project demonstrates a lightweight RAG architecture that:

1. Loads enterprise documentation
2. Splits documents into manageable chunks
3. Converts text into TF-IDF representations
4. Retrieves the most relevant chunks for a user query
5. Passes the retrieved context to an LLM
6. Generates a grounded response
7. Returns the answer together with the supporting source passages

### Architecture

```text
                    Enterprise Documents
                            │
                            ▼
                    Document Loading
                            │
                            ▼
                       Chunking
                            │
                            ▼
                 TF-IDF Text Representation
                            │
                            ▼
                  Similarity-based Retrieval
                            │
                            ▼
                    Top-k Relevant Chunks
                            │
                            ▼
                    Prompt Construction
                            │
                            ▼
                       Groq LLM
                            │
                            ▼
                 Grounded Answer Generation
                            │
                            ▼
                    FastAPI REST API
                            │
                            ▼
              Answer + Supporting Sources
