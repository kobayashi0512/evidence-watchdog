# Reproducibility protocol

## Fixed setting

Each instance already identifies one source document. The selector ranks only that document's candidate units:

- QASPER: full-text paragraphs paired with a question;
- SciFact: abstract sentences paired with a claim and an annotated evidence document.

This is a known-document evidence-selection experiment, not open-domain retrieval, answer generation, fact checking, or decision support. The candidate set and snippet budget are held constant within each comparison. Gold evidence never enters a ranking function.

## Metrics

- **QASPER evidence-matching score**: for each annotated evidence paragraph, find the maximum token-set F1 against any selected paragraph, then average over the annotated evidence paragraphs.
- **SciFact evidence-sentence recall**: fraction of annotated evidence sentence indices included in the selected set.
- **SciFact complete coverage**: whether all annotated evidence sentence indices are in the selected set.

The BGE-versus-BM25 intervals in `results/primary_results.json` are percentile paired cluster-bootstrap intervals from 20,000 resamples, seed `20260907`. QASPER resamples papers; SciFact resamples claims. Point estimates are observation-weighted. The release preserves the aggregate numeric artifact; it does not redistribute row-level benchmark-derived outputs. The manuscript's later unit-sensitivity checks are post hoc analyses derived from the official public records and frozen neural rankings.

## Execution order

1. Install dependencies: `pip install -r requirements.txt`.
2. Retrieve public assets: `python scripts/fetch_public_assets.py --datasets all --models all`.
3. Run a selected evaluation script with a local BGE or MiniLM path.
4. Compare your aggregate output with `results/primary_results.json`, accounting for upstream model revision and platform differences.

The original study repeatedly inspected development data and does not claim preregistered or confirmatory performance. It uses no model fine-tuning. Exact numeric reproduction requires the archived model files represented by `models/weights_manifest.json`; current public snapshots may differ.
