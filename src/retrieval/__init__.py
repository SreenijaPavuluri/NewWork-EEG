"""
EEG-RAG Retrieval Module
========================
Provides FAISS-based embedding store and natural language explanation
generation for the EEG-RAG system.
"""
from .store import EEGEmbeddingStore
from .explainer import generate_explanation, batch_explanation_summary

__all__ = ["EEGEmbeddingStore", "generate_explanation", "batch_explanation_summary"]
