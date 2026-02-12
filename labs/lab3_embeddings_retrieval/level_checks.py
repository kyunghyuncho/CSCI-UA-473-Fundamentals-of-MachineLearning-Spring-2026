import numpy as np


def check_step_1_loading(local_vars):
    """
    Step 1: Import SentenceTransformer and load the model.
    Expected: 'model' variable containing a SentenceTransformer instance.
    """
    if "SentenceTransformer" not in local_vars:
        return (
            False,
            "⚠️ Did you forget to import `SentenceTransformer` from `sentence_transformers`?",
        )

    if "model" not in local_vars:
        return (
            False,
            "⚠️ Variable `model` not found. Please assign the loaded model to a variable named `model`.",
        )

    model = local_vars["model"]
    # Check if it's the right type (relaxed check to avoid import issues)
    if not type(model).__name__ == "SentenceTransformer":
        return (
            False,
            f"⚠️ `model` seems to be {type(model)}, expected SentenceTransformer.",
        )

    return True, "✅ Model loaded successfully! You have the engine ready."


def check_step_2_encoding(local_vars):
    """
    Step 2: Encode text.
    Expected: 'embeddings' variable with shape (N, 768).
    """
    if "embeddings" not in local_vars:
        return False, "⚠️ Variable `embeddings` not found."

    embeddings = local_vars["embeddings"]

    # Check type
    if not hasattr(embeddings, "shape"):
        return (
            False,
            "⚠️ `embeddings` should be a numpy array or tensor (it doesn't have a shape property).",
        )

    # Check dimensionality
    # Nomic model is 768
    if len(embeddings.shape) < 2:
        return (
            False,
            f"⚠️ Embeddings should be a 2D matrix (batch_size, hidden_dim). Got shape {embeddings.shape}.",
        )

    if embeddings.shape[1] != 768:
        return (
            False,
            f"⚠️ Expected embedding dimension 768 (for Nomic model), got {embeddings.shape[1]}. Did you load the right model?",
        )

    return True, "✅ Text encoded! You've converted words to numbers."


def check_step_3_query_doc(local_vars):
    """
    Step 3: Query vs Document Embedding.
    Expected: 'query_emb' and 'doc_emb' variables.
    """
    if "query_emb" not in local_vars:
        return False, "⚠️ Variable `query_emb` not found."
    if "doc_emb" not in local_vars:
        return False, "⚠️ Variable `doc_emb` not found."

    q = local_vars["query_emb"]
    d = local_vars["doc_emb"]

    # Check shapes
    if not hasattr(q, "shape") or not hasattr(d, "shape"):
        return False, "⚠️ Embeddings must be numpy arrays or tensors."

    if q.shape[-1] != 768 or d.shape[-1] != 768:
        return False, "⚠️ Embeddings have wrong dimension (expected 768)."

    # Check if they actually used the prefixes (hard to check directly without inspecting code or values,
    # but we can check if they are identical if the text was similar - though text is likely different here).
    # We'll trust the user followed instructions or check code string if needed.
    # Actually, we can check if they are normalized.

    return True, "✅ Query and Document encoded separately!"


def check_step_4_similarity(local_vars):
    """
    Step 4: Calculate similarity.
    Expected: 'similarity' variable (float).
    """
    if "similarity" not in local_vars:
        return False, "⚠️ Variable `similarity` not found."

    sim = local_vars["similarity"]

    # Handle single element arrays/tensors
    if hasattr(sim, "item"):
        sim = sim.item()

    if not isinstance(sim, (float, int, np.floating, np.integer)):
        return False, f"⚠️ `similarity` should be a number, got {type(sim)}."

    # Check range (cosine similarity is [-1, 1])
    if sim < -1.1 or sim > 1.1:
        return (
            False,
            f"⚠️ Similarity score {sim} is outside the valid range [-1, 1]. Check your calculation (did you normalize?)",
        )

    return True, "✅ Similarity calculated! You can now measure meaning."


def check_step_5_knn(local_vars):
    """
    Step 5: K-Nearest Neighbors.
    Expected: 'top_k_indices' or 'top_k_scores'
    """
    if "scores" not in local_vars:
        return False, "⚠️ Variable `scores` (all similarity scores) not found."

    if "top_k_indices" not in local_vars:
        return False, "⚠️ Variable `top_k_indices` not found."

    local_vars["scores"]
    indices = local_vars["top_k_indices"]

    if len(indices) != 3:
        return False, f"⚠️ Expected 3 indices, got {len(indices)}."

    # Check if they are actually the top ones (roughly)
    # We can't strictly check exact indices without knowing their random corpus,
    # but we can check if indices are integers
    if not np.issubdtype(indices.dtype, np.integer):
        return False, "⚠️ Indices must be integers."

    return True, "✅ Search engine built! You found the needles in the haystack."
