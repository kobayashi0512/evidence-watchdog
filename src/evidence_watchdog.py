#!/usr/bin/env python3
"""Evidence Watchdog: a local evidence-first scientific reading assistant.

Input JSON:
  {"query": "...", "passages": ["...", "..."], "title": "optional"}

The tool ranks passages under a fixed selection budget and returns
source-context descriptors: whether selected material lies outside a
conventional leading view and whether it contains caution-related language.
They are structural source information, not alerts: the tool does not predict
omissions or assess truth, evidence sufficiency, or study quality.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

CAUTION = re.compile(
    r"\b(no|not|none|without|however|although|but|limit(?:ation|ed|s)?|uncertain|"
    r"contradict(?:s|ed|ory)?|inconsistent|cannot|caution|unlikely|failed|failure)\b",
    re.I,
)


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-9)


def rank(query: str, passages: list[str], model, tokenizer, device: str, backend: str, max_length: int, batch_size: int) -> list[float]:
    scores: list[float] = []
    model.eval()
    with torch.inference_mode():
        if backend == "embedding":
            q = tokenizer(query, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            q = {k: v.to(device) for k, v in q.items()}
            qvec = torch.nn.functional.normalize(mean_pool(model(**q).last_hidden_state, q["attention_mask"]), p=2, dim=1)
        for start in range(0, len(passages), batch_size):
            batch = passages[start : start + batch_size]
            if backend == "reranker":
                enc = tokenizer([query] * len(batch), batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            else:
                enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            if backend == "reranker":
                scores.extend(float(x) for x in model(**enc).logits.reshape(-1).detach().cpu().tolist())
            else:
                dvec = torch.nn.functional.normalize(mean_pool(model(**enc).last_hidden_state, enc["attention_mask"]), p=2, dim=1)
                scores.extend(float(x) for x in (dvec @ qvec.T).reshape(-1).detach().cpu().tolist())
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--backend", choices=["reranker", "embedding"], default="reranker")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--short-view", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    query = str(payload.get("query", "")).strip()
    passages = [str(x).strip() for x in payload.get("passages", []) if str(x).strip()]
    if not query or not passages:
        raise SystemExit("input must contain a non-empty query and passages list")
    if args.budget < 1 or args.short_view < 1:
        raise SystemExit("budget and short-view must be positive")

    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
    model_cls = AutoModelForSequenceClassification if args.backend == "reranker" else AutoModel
    model = model_cls.from_pretrained(str(args.model), local_files_only=True).to(device)
    scores = rank(query, passages, model, tokenizer, device, args.backend, args.max_length, args.batch_size)
    order = sorted(range(len(passages)), key=lambda i: (scores[i], -i), reverse=True)
    chosen = order[: min(args.budget, len(order))]
    outside = [i for i in chosen if i >= args.short_view]
    caution = [i for i in chosen if CAUTION.search(passages[i])]
    descriptors = []
    if outside:
        descriptors.append("Selected material lies outside the leading source view.")
    if caution:
        descriptors.append("Selected material contains caution-related wording.")
    if not descriptors:
        descriptors.append("No predefined source-context descriptor was present in the selected material.")

    result = {
        "title": payload.get("title"), "query": query, "model": str(args.model), "backend": args.backend,
        "budget": args.budget, "short_view_size": args.short_view,
        "top_evidence": [
            {"rank": rank_i + 1, "passage_index": idx, "score": scores[idx], "outside_short_view": idx >= args.short_view,
             "caution_terms": CAUTION.findall(passages[idx]), "text": passages[idx]}
            for rank_i, idx in enumerate(chosen)
        ],
        "source_context": {
            "selected_outside_leading_view": bool(outside),
            "selected_with_caution_language": bool(caution),
            "descriptors": descriptors,
            "scope": "Source-context descriptors identify structural display properties only; they do not predict omissions or assess truth, causality, evidence sufficiency, or study quality.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"top_indices": chosen, "source_context_descriptors": descriptors}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
