# Frozen primary results

`primary_results.json` is the single aggregate source for the manuscript's reported tables and figures. The values below are rounded for reading; use the JSON for full precision.

| Evaluation | Unit / metric | BGE at k=4 | BM25 at k=4 | Difference |
|---|---|---:|---:|---:|
| QASPER development | 89 questions; paragraph evidence-matching score | 0.562 | 0.396 | +0.165, 95% CI [0.083, 0.250] |
| SciFact development | 338 evidence sets; sentence recall | 0.788 | 0.663 | +0.125, 95% CI [0.078, 0.171] |
| SciFact development | complete evidence coverage | 77.2% | 64.2% | +13.0 percentage points |

The SciFact BGE k=4 view completely covered 261 of 338 annotated evidence sets and left 77 sets (22.8%) incomplete. This residual is the reason the interface reports an inspection cue and a route to the original source rather than a claim of evidence sufficiency.
