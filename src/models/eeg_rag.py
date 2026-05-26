"""
EEG-RAG: Retrieval-Augmented Motor Imagery Classification.

Architecture:
  1. STELLA Encoder (frozen pretrained) → query embedding (B, D)
  2. FAISS embedding store → retrieve K nearest neighbours
  3. RAGReasoningHead (cross-attention) → weighted aggregate → logits
  4. EEGRAGClassifier → full end-to-end pipeline

Key idea: Instead of classifying from the embedding alone, the model
conditions its prediction on K retrieved training examples (with known
labels), analogous to in-context learning in LLMs.

For a query embedding q and retrieved pairs {(e_i, y_i)}:
  - Label embeddings encode class identity
  - Cross-attention: q attends to {e_i + label_emb(y_i)}
  - Aggregated context + q → MLP → logits

This enables zero/few-shot generalization: even without retraining,
adding new labelled examples to the store improves performance.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .stella import STELLAEncoder, STELLAConfig
from ..retrieval.store import EEGEmbeddingStore


# ── RAG Reasoning Head ────────────────────────────────────────────────────────

class RAGReasoningHead(nn.Module):
    """
    Cross-attention reasoning head that conditions on retrieved examples.

    Input:
      query_emb   : (B, D) — current trial embedding from STELLA
      ret_embs    : (B, K, D) — retrieved neighbour embeddings
      ret_labels  : (B, K) — integer labels of retrieved neighbours

    Process:
      1. Encode retrieved labels into embeddings → add to ret_embs
      2. Cross-attention: query attends to labelled retrieved examples
      3. Prototype aggregation per class → class scores
      4. Concat [query, attended_context] → MLP → logits

    Output: (B, n_classes) logits
    """

    def __init__(
        self,
        d_model: int = 256,
        n_classes: int = 4,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes

        # Learnable label embeddings (one per class)
        self.label_embed = nn.Embedding(n_classes, d_model)

        # Cross-attention: query over retrieved examples
        self.cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        self.norm_q = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm_kv = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Class prototype aggregation
        # Weighted mean of retrieved embeddings per class
        self.proto_gate = nn.Linear(d_model, n_classes)

        # Final MLP: [query, context] → logits
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

        # Similarity-based direct vote (auxiliary)
        self.vote_weight = nn.Parameter(torch.tensor(0.3))   # learned mixing weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        query_emb: torch.Tensor,
        ret_embs: torch.Tensor,
        ret_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        query_emb  : (B, D)
        ret_embs   : (B, K, D) — already L2-normalised
        ret_labels : (B, K) long

        Returns
        -------
        logits : (B, n_classes)
        """
        B, K, D = ret_embs.shape

        # 1. Enrich retrieved embeddings with label context
        lbl_emb = self.label_embed(ret_labels)          # (B, K, D)
        kv = self.norm_kv[0](ret_embs + lbl_emb)        # (B, K, D)

        # 2. Cross-attention layers
        q = query_emb.unsqueeze(1)                       # (B, 1, D)
        for i, attn_layer in enumerate(self.cross_attn_layers):
            q_normed = self.norm_q[i](q)
            kv_normed = self.norm_kv[i](kv) if i > 0 else kv
            ctx, _ = attn_layer(q_normed, kv_normed, kv_normed)
            q = q + ctx                                  # residual

        context = q.squeeze(1)                           # (B, D)

        # 3. MLP classification
        combined = torch.cat([query_emb, context], dim=-1)  # (B, 2D)
        logits_mlp = self.classifier(combined)              # (B, n_classes)

        # 4. Soft majority vote from retrieved labels
        # Weight each retrieved example by cosine similarity
        sim = F.cosine_similarity(
            query_emb.unsqueeze(1).expand_as(ret_embs),
            ret_embs, dim=-1
        )                                                    # (B, K)
        sim_weights = F.softmax(sim, dim=-1)                 # (B, K)
        vote_logits = torch.zeros(B, self.n_classes, device=query_emb.device)
        for c in range(self.n_classes):
            mask = (ret_labels == c).float()                 # (B, K)
            vote_logits[:, c] = (sim_weights * mask).sum(-1)# (B,)

        # 5. Combine: learned mix of cross-attention logits + vote logits
        alpha = torch.sigmoid(self.vote_weight)
        logits = (1 - alpha) * logits_mlp + alpha * vote_logits

        return logits


# ── Full EEG-RAG Classifier ───────────────────────────────────────────────────

class EEGRAGClassifier(nn.Module):
    """
    Full EEG-RAG pipeline:
      STELLA encoder (frozen) → retrieval → RAGReasoningHead → logits.

    The store must be populated and indexed before inference.
    During N-shot adaptation: simply add new labelled examples to the
    store without any gradient updates — instant adaptation.

    Parameters
    ----------
    cfg : STELLAConfig
    n_classes : int
    k_neighbours : int  — how many examples to retrieve per query
    exclude_same_subject : bool — exclude same-subject neighbours (strict eval)
    """

    def __init__(
        self,
        cfg: STELLAConfig,
        n_classes: int = 4,
        k_neighbours: int = 5,
        exclude_same_subject: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = k_neighbours
        self.exclude_same_subject = exclude_same_subject
        self.n_classes = n_classes

        # Frozen STELLA encoder
        self.encoder = STELLAEncoder(cfg)

        # Cross-attention reasoning head (trainable)
        self.reasoning_head = RAGReasoningHead(
            d_model=cfg.d_model,
            n_classes=n_classes,
            n_heads=4,
            n_layers=2,
            dropout=dropout,
        )

        # Embedding store (populated externally)
        self.store: EEGEmbeddingStore | None = None

    def set_store(self, store: EEGEmbeddingStore) -> None:
        """Attach a populated FAISS embedding store."""
        self.store = store

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode EEG trials, return L2-normalised embeddings."""
        emb = self.encoder(x)
        return F.normalize(emb, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        subject_ids: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, T) EEG trials
        subject_ids : list of subject IDs per sample (for exclusion)

        Returns
        -------
        logits : (B, n_classes)
        """
        assert self.store is not None, "Call set_store() before forward()"

        B = x.shape[0]
        device = x.device

        # Encode query
        query_emb = self.encode(x)                          # (B, D)

        # Retrieve K neighbours per sample
        ret_embs_list, ret_lbls_list = [], []
        for i in range(B):
            q_np = query_emb[i:i+1].cpu().numpy()
            sid = subject_ids[i] if subject_ids and self.exclude_same_subject else None
            _, r_embs, r_lbls, _ = self.store.search(q_np, k=self.k,
                                                       exclude_subject=sid)
            ret_embs_list.append(r_embs[0])                 # (K, D)
            ret_lbls_list.append(r_lbls[0])                 # (K,)

        ret_embs = torch.tensor(
            np.stack(ret_embs_list), dtype=torch.float32, device=device
        )                                                    # (B, K, D)
        ret_lbls = torch.tensor(
            np.stack(ret_lbls_list), dtype=torch.long, device=device
        )                                                    # (B, K)

        # Reasoning head
        logits = self.reasoning_head(query_emb, ret_embs, ret_lbls)
        return logits

    def predict_with_explanation(
        self,
        x: torch.Tensor,
        subject_id: int | None = None,
    ) -> tuple[int, float, str]:
        """
        Single-sample prediction with natural language explanation.

        Returns
        -------
        predicted_class : int
        confidence : float
        explanation : str
        """
        from ..retrieval.explainer import generate_explanation

        assert x.ndim == 2, "Expected (C, T) single trial — no batch dim"
        x_batch = x.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            q_emb = self.encode(x_batch)
            q_np = q_emb[0].cpu().numpy()
            sid_excl = subject_id if self.exclude_same_subject else None
            dists, r_embs, r_lbls, r_sids = self.store.search(
                q_np[np.newaxis], k=self.k, exclude_subject=sid_excl
            )
            r_embs_t = torch.tensor(r_embs, dtype=torch.float32)
            r_lbls_t = torch.tensor(r_lbls, dtype=torch.long)
            logits = self.reasoning_head(q_emb, r_embs_t, r_lbls_t)
            probs = F.softmax(logits, dim=-1)[0].cpu().numpy()

        pred_class = int(probs.argmax())
        confidence = float(probs[pred_class])

        explanation = generate_explanation(
            predicted_class=pred_class,
            confidence=confidence,
            retrieved_labels=r_lbls[0],
            retrieved_distances=dists[0],
            subject_id=subject_id,
        )
        return pred_class, confidence, explanation


# ── Prototype-only baseline (no training needed) ──────────────────────────────

class PrototypeClassifier:
    """
    Simple prototype (centroid) classifier using STELLA embeddings.
    No training required — just compute class centroids from labelled examples.
    Used as a sub-baseline for N-shot evaluation.
    """

    def __init__(self, n_classes: int = 4):
        self.n_classes = n_classes
        self.prototypes: np.ndarray | None = None  # (n_classes, D)

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        D = embeddings.shape[1]
        self.prototypes = np.zeros((self.n_classes, D), dtype=np.float32)
        for c in range(self.n_classes):
            mask = labels == c
            if mask.sum() > 0:
                proto = embeddings[mask].mean(0)
                self.prototypes[c] = proto / (np.linalg.norm(proto) + 1e-8)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.prototypes is not None
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embs_norm = embeddings / norms
        scores = embs_norm @ self.prototypes.T               # (N, n_classes)
        return scores.argmax(axis=1)

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        assert self.prototypes is not None
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embs_norm = embeddings / norms
        scores = embs_norm @ self.prototypes.T
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)
