# Data availability and provenance

## Released by this repository

This repository releases source code, synthetic-format examples, a figure deck, a model-provenance manifest, and aggregate derived results used by the manuscript. `results/primary_results.json` contains no source-document text, questions, claims, or individual evidence labels.

## Obtained from original public providers

| Resource | Study use | Official project | Retrieval route used by `scripts/fetch_public_assets.py` |
|---|---|---|---|
| QASPER v0.3 | Paragraph-level evidence visibility on full scientific papers | https://github.com/allenai/qasper-led-baseline | https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz |
| SciFact | Abstract-sentence evidence visibility for scientific claims | https://github.com/allenai/scifact | https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz |
| BGE reranker | Fixed pretrained cross-encoder selector; not fine-tuned | https://huggingface.co/BAAI/bge-reranker-v2-m3 | Hugging Face snapshot download |
| MiniLM | Fixed pretrained bi-encoder selector; not fine-tuned | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Hugging Face snapshot download |

The QASPER and SciFact archives are public research releases and were chosen specifically because they require no local ethics approval or access to restricted data. Users are responsible for accepting and following the terms, licences, and citation requirements set by the data and model providers.

## What is not redistributed

The project does not redistribute QASPER papers, questions, evidence annotations, SciFact abstracts/claims/rationales, BGE checkpoint files, or MiniLM checkpoint files. Those materials remain under upstream control. Download them directly with the helper script or from the official links above.

## Study boundary

The datasets provide human-provided evidence annotations for evaluating whether a short, ranked view makes those annotations visible. They do not establish evidence sufficiency, field-level scientific bias, or user outcomes.
