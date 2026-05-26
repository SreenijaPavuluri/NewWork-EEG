"""
FAISS-based EEG Embedding Store for EEG-RAG.

Stores STELLA encoder embeddings indexed by subject and label,
enabling efficient K-nearest-neighbour retrieval at inference time.
"""

from __future__ import annotations
import json
import pickle
from pathlib import Path
import numpy as np
import torch

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class EEGEmbeddingStore:
    """
    FAISS-indexed store of EEG trial embeddings produced by STELLA encoder.

    Usage
    -----
    store = EEGEmbeddingStore(d_model=256)
    store.add(embeddings, labels, subject_ids)   # build index
    dists, embs, lbls, sids = store.search(query, k=5)
    store.save("results/rag_store")
    store2 = EEGEmbeddingStore.load("results/rag_store")
    """

    def __init__(self, d_model: int = 256, use_cosine: bool = True):
        self.d_model = d_model
        self.use_cosine = use_cosine
        self._embeddings: list[np.ndarray] = []
        self._labels: list[int] = []
        self._subject_ids: list[int] = []
        self._index = None
        self._built = False

    # ── population ────────────────────────────────────────────────────────────

    def add(
        self,
        embeddings: torch.Tensor | np.ndarray,
        labels: torch.Tensor | list[int],
        subject_ids: torch.Tensor | list[int],
    ) -> None:
        """Add a batch of embeddings to the store (does not rebuild index)."""
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.detach().cpu().float().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().tolist()
        if isinstance(subject_ids, torch.Tensor):
            subject_ids = subject_ids.cpu().tolist()

        self._embeddings.append(embeddings)
        self._labels.extend(labels)
        self._subject_ids.extend(subject_ids)
        self._built = False

    def build_index(self) -> None:
        """Concatenate all added embeddings and build FAISS index."""
        all_embs = np.concatenate(self._embeddings, axis=0).astype(np.float32)

        if self.use_cosine:
            # L2-normalize for cosine similarity via inner product
            norms = np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8
            all_embs = all_embs / norms

        if FAISS_AVAILABLE:
            if self.use_cosine:
                self._index = faiss.IndexFlatIP(self.d_model)  # inner product = cosine after norm
            else:
                self._index = faiss.IndexFlatL2(self.d_model)
            self._index.add(all_embs)
        # Always keep raw numpy array for fallback and metadata lookup
        self._all_embs = all_embs
        self._all_labels = np.array(self._labels, dtype=np.int64)
        self._all_sids = np.array(self._subject_ids, dtype=np.int64)
        self._built = True

    # ── retrieval ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: torch.Tensor | np.ndarray,
        k: int = 5,
        exclude_subject: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve K nearest neighbours for query embedding(s).

        Parameters
        ----------
        query : (D,) or (B, D)
        k : number of neighbours to retrieve
        exclude_subject : if set, exclude trials from this subject ID

        Returns
        -------
        distances  : (B, k) similarity scores
        embeddings : (B, k, D)
        labels     : (B, k) integer class labels
        subject_ids: (B, k)
        """
        if not self._built:
            self.build_index()

        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().float().numpy()
        if query.ndim == 1:
            query = query[np.newaxis]  # (1, D)

        if self.use_cosine:
            norms = np.linalg.norm(query, axis=1, keepdims=True) + 1e-8
            query_norm = query / norms
        else:
            query_norm = query

        B = query_norm.shape[0]

        if exclude_subject is not None:
            # Slow fallback: manual masked search
            return self._masked_search(query_norm, k, exclude_subject)

        if FAISS_AVAILABLE:
            # Retrieve more to allow post-filtering if needed
            k_fetch = min(k, self._all_embs.shape[0])
            dists, idxs = self._index.search(query_norm.astype(np.float32), k_fetch)
        else:
            # Pure numpy cosine similarity fallback
            scores = query_norm @ self._all_embs.T  # (B, N)
            idxs = np.argsort(-scores, axis=1)[:, :k]
            dists = np.take_along_axis(scores, idxs, axis=1)

        ret_embs = self._all_embs[idxs]           # (B, k, D)
        ret_lbls = self._all_labels[idxs]          # (B, k)
        ret_sids = self._all_sids[idxs]            # (B, k)

        return dists, ret_embs, ret_lbls, ret_sids

    def _masked_search(self, query_norm, k, exclude_sid):
        """Search excluding all trials from a given subject."""
        mask = self._all_sids != exclude_sid
        cand_embs = self._all_embs[mask]
        cand_lbls = self._all_labels[mask]
        cand_sids = self._all_sids[mask]

        scores = query_norm @ cand_embs.T           # (B, N_masked)
        k_actual = min(k, cand_embs.shape[0])
        idxs = np.argsort(-scores, axis=1)[:, :k_actual]
        dists = np.take_along_axis(scores, idxs, axis=1)

        ret_embs = cand_embs[idxs]
        ret_lbls = cand_lbls[idxs]
        ret_sids = cand_sids[idxs]
        return dists, ret_embs, ret_lbls, ret_sids

    # ── class prototypes ─────────────────────────────────────────────────────

    def class_prototypes(self, n_classes: int = 4) -> np.ndarray:
        """Return mean embedding per class. Shape: (n_classes, D)."""
        if not self._built:
            self.build_index()
        protos = np.zeros((n_classes, self.d_model), dtype=np.float32)
        for c in range(n_classes):
            mask = self._all_labels == c
            if mask.sum() > 0:
                protos[c] = self._all_embs[mask].mean(0)
        return protos

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save store to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if not self._built:
            self.build_index()
        np.save(path / "embeddings.npy", self._all_embs)
        np.save(path / "labels.npy", self._all_labels)
        np.save(path / "subject_ids.npy", self._all_sids)
        meta = {"d_model": self.d_model, "use_cosine": self.use_cosine,
                "n_trials": len(self._all_labels)}
        (path / "meta.json").write_text(json.dumps(meta, indent=2))
        if FAISS_AVAILABLE:
            faiss.write_index(self._index, str(path / "faiss.index"))
        print(f"[Store] Saved {len(self._all_labels):,} embeddings → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "EEGEmbeddingStore":
        """Load store from directory."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        store = cls(d_model=meta["d_model"], use_cosine=meta["use_cosine"])
        store._all_embs = np.load(path / "embeddings.npy")
        store._all_labels = np.load(path / "labels.npy")
        store._all_sids = np.load(path / "subject_ids.npy")
        store._embeddings = [store._all_embs]
        store._labels = store._all_labels.tolist()
        store._subject_ids = store._all_sids.tolist()
        if FAISS_AVAILABLE and (path / "faiss.index").exists():
            store._index = faiss.read_index(str(path / "faiss.index"))
        store._built = True
        print(f"[Store] Loaded {len(store._all_labels):,} embeddings ← {path}")
        return store

    def __len__(self) -> int:
        return len(self._labels)

    def __repr__(self) -> str:
        return (f"EEGEmbeddingStore(n={len(self)}, d={self.d_model}, "
                f"faiss={'yes' if FAISS_AVAILABLE else 'numpy-fallback'})")
