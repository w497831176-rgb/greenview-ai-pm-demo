"""
Local text embedding for RAG.

Primary: BAAI/bge-small-zh-v1.5 (512 dims) loaded from local HuggingFace cache.
Fallback: deterministic character-ngram embedding when the model is not cached and
HuggingFace is unreachable (e.g., NAS network restrictions).
"""

import hashlib
import math
import os
import re
from pathlib import Path
from typing import List, Optional

MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
CACHE_DIR = Path(os.getenv("PROPERTY_DATA_DIR", "/app/data")) / "models"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DIM = 512

_tokenizer = None
_model = None
_use_fallback = None


def _model_cached() -> bool:
    """Check whether the transformer model files exist locally."""
    model_cache = CACHE_DIR / f"models--{MODEL_NAME.replace('/', '--')}"
    if model_cache.exists():
        return True
    for marker in ["config.json", "pytorch_model.bin", "model.safetensors"]:
        if (CACHE_DIR / MODEL_NAME / marker).exists():
            return True
    return False


def _load_transformer_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        from transformers import AutoModel, AutoTokenizer
        # Prefer a flat local copy under <data>/models/<MODEL_NAME>/ if it exists.
        local_path = CACHE_DIR / MODEL_NAME
        if local_path.exists() and (local_path / "config.json").exists():
            load_path = str(local_path)
        else:
            load_path = MODEL_NAME
        _tokenizer = AutoTokenizer.from_pretrained(load_path, cache_dir=str(CACHE_DIR), local_files_only=True)
        _model = AutoModel.from_pretrained(load_path, cache_dir=str(CACHE_DIR), local_files_only=True)
        _model.eval()
    return _tokenizer, _model


def _embed_transformer(text: str) -> List[float]:
    import torch
    tokenizer, model = _load_transformer_model()
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)

    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * input_mask_expanded).sum(dim=1)
    embeddings = sum_embeddings / input_mask_expanded.sum(dim=1).clamp(min=1e-9)
    return embeddings[0].tolist()


def _tokenize_for_fallback(text: str) -> List[str]:
    """Extract character n-grams and word-like tokens."""
    text = text.strip().lower()
    # Remove common Chinese question/stop filler characters but keep content words.
    text = re.sub(r"[？?。！!,，.。;；：:\s]+", " ", text)
    tokens = []

    # Word-level tokens split by spaces (preserves multi-char terms after punctuation cleaning)
    for word in text.split():
        if word:
            tokens.append(word)
            # Sub-word n-grams for multi-character words
            for n in (2, 3, 4):
                if len(word) >= n:
                    for i in range(len(word) - n + 1):
                        tokens.append(word[i:i + n])

    # Character-level n-grams over the whole normalized string (without spaces)
    compact = text.replace(" ", "")
    for n in (1, 2, 3):
        for i in range(len(compact) - n + 1):
            tokens.append(compact[i:i + n])

    return tokens


def _embed_fallback(text: str) -> List[float]:
    """
    Deterministic character/word n-gram embedding.
    Builds a 512-dim vector by hashing n-grams to positions.
    """
    vec = [0.0] * VECTOR_DIM
    if not text:
        return vec

    tokens = _tokenize_for_fallback(text)
    if not tokens:
        return vec

    # TF weighting: rare tokens get slightly higher weight
    from collections import Counter
    counts = Counter(tokens)
    total = len(tokens)

    for token in tokens:
        h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
        pos1 = h % VECTOR_DIM
        pos2 = (h >> 9) % VECTOR_DIM
        pos3 = (h >> 18) % VECTOR_DIM
        # TF * log(1 + length) weighting
        tf = counts[token] / total
        weight = tf * (1.0 + math.log1p(len(token)))
        vec[pos1] += weight
        vec[pos2] += weight * 0.7
        vec[pos3] += weight * 0.4

    # L2 normalize
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _should_use_fallback() -> bool:
    global _use_fallback
    if _use_fallback is None:
        _use_fallback = not _model_cached()
        if _use_fallback:
            print("[rag_embeddings] Transformer model not cached; using deterministic fallback embedding.")
        else:
            print("[rag_embeddings] Using cached transformer model.")
    return _use_fallback


def embed_text(text: str) -> List[float]:
    """Embed a single text into a dense 512-dim vector."""
    if _should_use_fallback():
        return _embed_fallback(text)
    return _embed_transformer(text)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts."""
    return [embed_text(t) for t in texts]


def get_runtime_info() -> dict:
    """Return an honest, serialisable description of the semantic channel."""
    fallback = _should_use_fallback()
    return {
        "model_name": MODEL_NAME,
        "mode": "deterministic_ngram_fallback" if fallback else "local_transformer",
        "label": "确定性 n-gram 向量降级" if fallback else "本地 Transformer 向量模型",
        "is_semantic_model": not fallback,
        "note": (
            "当前未缓存本地 Transformer，语义通道使用确定性 n-gram 向量；"
            "可演示混合检索流程，但不应表述为已启用预训练语义模型。"
            if fallback else "向量通道使用本地缓存的 Transformer 模型，不调用聊天模型。"
        ),
    }
