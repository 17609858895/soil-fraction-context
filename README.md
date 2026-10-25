# Conditional prediction of dominant soil metal fractions

Core scientific code for *Co-measured elements improve dominant metal fraction prediction in sparse soil records*.

The analysis tests whether observed element–fraction pairs improve inference of another element's dominant operational extraction fraction. F1–F4 are acid-soluble, reducible, oxidizable and residual dominance classes. This repository contains analysis code, not manuscript files, precomputed predictions or source data.

## Data

The public source is *Classification of soil heavy metal fraction*, Figshare version 5:
https://doi.org/10.6084/m9.figshare.29554202.v5 (CC BY 4.0).
Run `python fetch_data.py` to download and checksum-verify its 546,799-byte modelling table. This source contains 9,489 records and 52 recovered element identities. Preserve its row order: reconstructed source blocks and repeated-cell handling use that order.

## Environment and main workflow

Use Python 3.11 in an isolated environment, then install `requirements.txt`.
The dependency ranges describe compatible APIs, not an exact lock of every historical training environment. Numerical equivalence can depend on library versions, threading and hardware, particularly for neural-network training.

```text
python -m pip install -r requirements.txt
python fetch_data.py
python run_core.py --plan
python run_core.py
```

The core runner explicitly sets `BASE_LEARNER=RF` for the nested stack, matching the reported principal random-forest comparison. Do not leave the stack at its auto-selection default when reproducing that comparison. Output goes to `analysis/results/`. Training is substantial; the runner fits models and is not a quick demonstration.

## Analyses and dependencies

| Programs | Scientific role / execution dependency |
|---|---|
| hm_data, hm_geochem | Data, grouping, elemental descriptors and metrics shared by all analyses |
| 01 | Twenty-classifier reference benchmark; run before 14 |
| 02–04 | Sequential-class, element-transfer and pH association analyses |
| 05, 05b | Prediction-set comparisons; 05b is the separated temperature/calibration analysis |
| 06 | Context representations, substitution and sentinel comparisons; supplies outputs for 09 |
| 07–08 | Masked set model and unseen-element comparisons |
| 09 | Residual co-fractionation; run after 06 |
| 10–11 | Random visibility and twenty-classifier context comparisons |
| 12 | pH sensitivity checks |
| 13 | Nested conditional forest/set-model stack; specify RF for the principal comparison |
| 14 | Repeated folds and nested tuning; run after 01 |
| 15–16 | Reconstructed source-block analyses; run 15 before 16 |
| 17 | Completion comparators and removal of target total content; run after 13 with RF |
| 18 | Retrospective measurement ordering |
| 19 | Priority-metal completion and uncertainty; run after 17 |

Each program runs as `python analysis/scripts/<program>.py`. Program numbers identify code files, not manuscript figures. The exploratory random-split templates and document/layout tooling are excluded.

## Interpretation and verification

Soil-property groups are exact matches on four released covariates; original sample and study identifiers are unavailable. They are proxies for co-measurement. The code excludes all labels of the target element from inference context. Some comparative procedures have their own preprocessing and transductive rules, described in the Supporting Information; this is not an external-study validation pipeline.

The principal grouped comparison reports macro F1 0.640 without context and 0.731 with context. The completion model omits unknown target total content and produces 10,948 missing-cell predictions for seven metals. These missing-cell predictions are not new measurements.

For this code release, source syntax, execution dependency order, data checksum, grouping and target exclusion were checked. The full training suite was not rerun for publication of the repository. All numerical claims refer to the study's completed analyses.
