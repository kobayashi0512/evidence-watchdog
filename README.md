# Evidence Watchdog

**Evidence Watchdog** is a reproducibility package for the fixed-budget evidence-visibility study, *AI Makes More Scientific Evidence Visible but Leaves Some Evidence Out of Sight*.

Scientific reading interfaces often replace a full document with a few ranked snippets. This package asks a narrow, auditable question: **under the same small snippet budget, how much human-annotated evidence remains visible?** It does not estimate truth, causality, study quality, or clinical benefit.

The included prototype ranks candidate passages, returns the original text and its source position, and raises a review cue when selected material sits outside the conventional short view or contains caution-related language. The cue is intentionally heuristic: it is a prompt to inspect the source, not a detector of a particular omitted sentence.

## What is released

| Item | Location | Purpose |
|---|---|---|
| Source-inspection prototype | `src/evidence_watchdog.py` | Rank passages with a local BGE reranker or MiniLM bi-encoder and return source-aware output. |
| Evaluation scripts | `scripts/` | Reproduce the document-local QASPER and SciFact neural-selector evaluations. |
| Frozen aggregate results | `results/primary_results.json` | Numeric source for the paper's tables, confidence intervals, and coverage analysis. |
| Example inputs and outputs | `examples/` | Small, runnable format examples without redistributing benchmark corpora. |
| Editable figures | `figures/` | PowerPoint source used for the manuscript figures. |
| Model manifest and fetcher | `models/`, `scripts/fetch_public_assets.py` | Provenance and local download path for public, third-party checkpoints. |

The repository deliberately does **not** redistribute QASPER, SciFact, or third-party model checkpoints. Download them from their original providers under their respective terms.

## Main paper results

All numbers below are retrospective, development-partition evidence-visibility measurements; they are not clinical, causal, or answer-accuracy results.

| Dataset and fixed budget | BGE | Document-local BM25 | Paired BGE gain (95% cluster-bootstrap interval) |
|---|---:|---:|---:|
| QASPER, top-4 paragraphs | 0.562 evidence-matching score | 0.396 | +0.165 [0.083, 0.250] |
| SciFact, top-4 sentences | 0.788 evidence-sentence recall | 0.663 | +0.125 [0.078, 0.171] |

On SciFact, complete coverage rose from 64.2% with document-local BM25 to 77.2% with BGE at four sentences; 22.8% of annotated evidence sets still remained incomplete. Thus, a higher ranking score does not certify evidence sufficiency.

## Quick start

Create an environment and install the package requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/fetch_public_assets.py --models bge-reranker-v2-m3
```

Run the BGE source-inspection example:

```bash
python src/evidence_watchdog.py \
  --input examples/scifact_input.json \
  --output outputs/scifact_watchdog.json \
  --model models/bge-reranker-v2-m3 \
  --backend reranker --budget 4 --short-view 2
```

Use `--models all-MiniLM-L6-v2` with the fetcher and `--backend embedding` for the MiniLM example.

## Data provenance and access

No approval-controlled, clinical, patient, or participant-level data are used in this release.

- **QASPER v0.3**: information-seeking questions and paragraph-level evidence anchored in research papers. Official project: [allenai/qasper-led-baseline](https://github.com/allenai/qasper-led-baseline). The original v0.3 train/dev archive is retrieved by the supplied script from `https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz`.
- **SciFact**: scientific claims, abstracts, and sentence-level rationale annotations. Official project: [allenai/scifact](https://github.com/allenai/scifact). The supplied script retrieves its official release archive from `https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`.

The benchmark data remain subject to their creators' notices and licences. This repository stores only aggregate, derived numbers and format examples; it does not copy benchmark documents, questions, claims, or annotations into the release.

Run:

```bash
python scripts/fetch_public_assets.py --datasets all
```

to place the archives and extracted files under `data/`. See [docs/DATA_AVAILABILITY.md](docs/DATA_AVAILABILITY.md) for the exact scope and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the evaluation protocol.

## Model weights and provenance

No checkpoint was trained or fine-tuned for this study. The experiments used publicly released third-party checkpoints:

- `BAAI/bge-reranker-v2-m3` as the cross-encoder reranker;
- `sentence-transformers/all-MiniLM-L6-v2` as the bi-encoder.

`models/weights_manifest.json` records the archived local experiment fingerprints. `scripts/fetch_public_assets.py` downloads the public sources into `models/` and prints checksums of the retrieved files. Consult each upstream model card for licence, revision history, and usage terms; this repository's MIT licence does not cover those model files.

## Reproducing the evaluations

The scripts evaluate a **known-document** setting only: for a question/claim paired with one document, candidate paragraphs or abstract sentences are ranked under fixed budgets of 2 and 4. Gold evidence is used only after selection for evaluation.

```bash
# QASPER: all answerable yes/no development questions in the paper's robustness diagnostic
python scripts/qasper_semantic_visibility_pilot.py \
  --input data/qasper/qasper-dev-v0.3.json \
  --model models/bge-reranker-v2-m3 \
  --backend reranker --all-yesno \
  --output outputs/qasper_bge_dev.json

# SciFact: document-local rationale recovery on the development partition
python scripts/scifact_semantic_visibility_pilot.py \
  --claims data/scifact/data/claims_dev.jsonl \
  --corpus data/scifact/data/corpus.jsonl \
  --model models/bge-reranker-v2-m3 \
  --backend reranker \
  --output outputs/scifact_bge_dev.json
```

Paths may differ slightly with future upstream archive layouts; inspect `data/` after extraction. The `results/primary_results.json` file is the frozen aggregate source for the paper, including the 20,000-resample paired cluster-bootstrap results. Re-running a current upstream model release without the archived experiment revision can yield different values.

## Scope and responsible use

Evidence Watchdog is a literature-navigation and source-inspection prototype. It does not make diagnoses, recommend treatments, judge study validity, or establish that a scientific field has been biased. A highlighted source position is a route for human inspection, not a statement that omitted evidence is false, unimportant, or known.

## Licence

The code and original documentation in this repository are released under the [MIT License](LICENSE). Dataset and model licences remain with their respective providers.
