"""
Natural language explanation generator for EEG-RAG predictions.

Generates structured, interpretable text explanations combining:
- Retrieval evidence (K nearest neighbours + their labels)
- Confidence from the reasoning head
- Band-power context from the spectral stream
- Clinical language referencing known MI literature

This is the "language generation" component of EEG-RAG.
"""

from __future__ import annotations
import numpy as np

CLASS_NAMES = {
    0: "Left Fist",
    1: "Right Fist",
    2: "Both Fists",
    3: "Both Feet",
}

LATERALITY = {
    0: "left hemisphere contralateral activation (C4 alpha suppression)",
    1: "right hemisphere contralateral activation (C3 alpha suppression)",
    2: "bilateral sensorimotor activation (C3 and C4 alpha suppression)",
    3: "central midline activation (Cz, bilateral leg representation)",
}

BAND_CONTEXT = {
    0: "alpha (8-13 Hz) ERD dominant at C4",
    1: "alpha (8-13 Hz) ERD dominant at C3",
    2: "bilateral alpha ERD with beta (13-30 Hz) rebound",
    3: "mu (8-12 Hz) and beta (20-30 Hz) bilateral ERD at Cz",
}


def generate_explanation(
    predicted_class: int,
    confidence: float,
    retrieved_labels: np.ndarray,
    retrieved_distances: np.ndarray,
    subject_id: int | None = None,
    query_band_powers: np.ndarray | None = None,
) -> str:
    """
    Generate a structured natural language explanation for a RAG prediction.

    Parameters
    ----------
    predicted_class : int — predicted MI class (0-3)
    confidence : float — softmax confidence for predicted class
    retrieved_labels : (K,) — labels of retrieved nearest neighbours
    retrieved_distances : (K,) — cosine similarity scores of retrieved examples
    subject_id : optional subject identifier
    query_band_powers : optional (5,) band powers [delta, theta, alpha, beta, gamma]

    Returns
    -------
    explanation : str — multi-line natural language explanation
    """
    K = len(retrieved_labels)
    pred_name = CLASS_NAMES.get(predicted_class, f"Class {predicted_class}")
    conf_pct = confidence * 100

    # ── Retrieval evidence ────────────────────────────────────────────────────
    vote_counts = {c: 0 for c in range(4)}
    for lbl in retrieved_labels:
        vote_counts[int(lbl)] += 1
    majority_class = max(vote_counts, key=vote_counts.get)
    majority_name = CLASS_NAMES.get(majority_class, f"Class {majority_class}")
    majority_votes = vote_counts[majority_class]

    retrieval_lines = []
    for i, (lbl, dist) in enumerate(zip(retrieved_labels, retrieved_distances)):
        lbl_name = CLASS_NAMES.get(int(lbl), f"Cls{lbl}")
        retrieval_lines.append(
            f"    [{i+1}] {lbl_name:<12}  similarity={dist:.3f}"
        )

    # ── Band power context ────────────────────────────────────────────────────
    band_lines = ""
    if query_band_powers is not None:
        band_names = ["δ(delta)", "θ(theta)", "α(alpha)", "β(beta)", "γ(gamma)"]
        bp_str = "  ".join(
            f"{n}={v:.2f}" for n, v in zip(band_names, query_band_powers)
        )
        band_lines = f"\n  Band powers : {bp_str}"

    # ── Confidence descriptor ────────────────────────────────────────────────
    if conf_pct >= 75:
        conf_desc = "HIGH"
    elif conf_pct >= 55:
        conf_desc = "MODERATE"
    else:
        conf_desc = "LOW"

    # ── Agreement check ──────────────────────────────────────────────────────
    if majority_class == predicted_class:
        agreement = f"✓ Majority of retrieved examples agree ({majority_votes}/{K})"
    else:
        agreement = (
            f"△ Retrieval majority ({majority_name}, {majority_votes}/{K}) "
            f"differs from model prediction — review recommended"
        )

    # ── Compose explanation ───────────────────────────────────────────────────
    subject_str = f"Subject S{subject_id:03d}" if subject_id else "Query trial"
    explanation = f"""
╔══════════════════════════════════════════════════════════════════════╗
║  EEG-RAG Prediction Report                                          ║
╚══════════════════════════════════════════════════════════════════════╝
  Trial        : {subject_str}
  Prediction   : {pred_name} (class {predicted_class})
  Confidence   : {conf_pct:.1f}%  [{conf_desc}]
  Neural basis : {LATERALITY.get(predicted_class, "unknown")}
  EEG signature: {BAND_CONTEXT.get(predicted_class, "unknown")}
{band_lines}

  Retrieval Evidence ({K} nearest neighbours from training set):
{chr(10).join(retrieval_lines)}

  {agreement}

  Interpretation:
    The STELLA encoder mapped this trial to an embedding region predominantly
    occupied by {pred_name} examples. The cross-attention reasoning head
    weighted retrieved neighbours by cosine similarity ({retrieved_distances.max():.3f} max)
    and produced a {conf_desc.lower()}-confidence prediction.
{"    ⚠ Low confidence — consider requesting an additional trial." if conf_pct < 50 else ""}
""".strip()

    return explanation


def batch_explanation_summary(
    predicted_classes: list[int],
    confidences: list[float],
    true_classes: list[int] | None = None,
) -> str:
    """Generate a batch summary table for multiple predictions."""
    lines = ["EEG-RAG Batch Prediction Summary", "=" * 50]
    correct = 0
    for i, (pred, conf) in enumerate(zip(predicted_classes, confidences)):
        pred_name = CLASS_NAMES.get(pred, str(pred))
        row = f"  Trial {i+1:3d}: {pred_name:<12} conf={conf*100:.1f}%"
        if true_classes is not None:
            true_name = CLASS_NAMES.get(true_classes[i], str(true_classes[i]))
            match = "✓" if pred == true_classes[i] else "✗"
            row += f"  gt={true_name:<12} {match}"
            correct += (pred == true_classes[i])
        lines.append(row)

    if true_classes is not None:
        acc = correct / max(len(predicted_classes), 1)
        lines.append("-" * 50)
        lines.append(f"  Accuracy: {correct}/{len(predicted_classes)} = {acc:.4f}")
    return "\n".join(lines)
