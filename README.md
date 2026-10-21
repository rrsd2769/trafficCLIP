# OptimizedTrafficCLIP: Reproducibility Package

Code, corrected pipelines, and per-seed results for an independent reproducibility study of
OptimizedTrafficCLIP, a tri-modal encrypted-traffic classifier. The paper is
`paper/OptimizedTrafficCLIP_IEEE.tex` (compiled: `paper/OptimizedTrafficCLIP_IEEE.pdf`).

The study runs five stages against one shared data foundation and one shared test partition:

| Stage | Question | Where |
|---|---|---|
| S1 | Does the released code implement the stated protocol? | Defect audit, `src/` (fixes in place) |
| S2 | What does the corrected release score? | Track B, `src/`, `scripts/run_b_*.py` |
| S3 | Is the gap architectural or implementation-specific? | Tracks A and A-Full, `A/` |
| S4 | How hard is the task, and does the finding transfer? | Track C, `C/`, three extra datasets |
| S5 | Where does the residual error come from? | `scripts/analyze_confusable_pair.py` and friends |

## Repository map

| Path | Contents |
|---|---|
| `A/` | Track A, independent bi-modal reimplementation of TrafficCLIP (`train.py`, `model.py`). Track A-Full, independent tri-modal rebuild of OptimizedTrafficCLIP (`*_full.py`). |
| `src/` | Track B, the released OptimizedTrafficCLIP implementation with its six defects fixed in place. Also the shared dataset builder, `src/make_traffic_dataset.py`. |
| `C/` | Track C, four from-scratch baselines: FlowPic, FS-Net, 1D-CNN, Transformer (`C/baselines/`). `run_matrix.py` trains them, `make_tables.py` builds the master table. |
| `scripts/` | Dataset download and extraction, experiment runners, statistics, and plotting. |
| `results/comparison/` | Every per-seed and aggregated CSV behind the paper's tables. |
| `results/figures/` | Every figure, at 300 DPI. |
| `tests/`, `A/tests/`, `C/tests/` | Pipeline, cache-equivalence, and split-identity tests. |
| `paper/` | Manuscript source, compiled PDF, and the figures it includes. |
| `config/config.yaml` | Track B configuration. |

All tracks use the same stratified 70/15/15 split at seed 62. `C/tests/test_split_identity.py`
asserts that the split code of each track produces identical test indices for the same archive.

## Setup

Python 3.12. From the project root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-tracking.txt
```

`requirements-tracking.txt` includes `requirements.txt` plus MLflow and ONNX, which Track B
needs. `requirements-lock.txt` lists the exact versions of the tested environment (macOS arm64,
Apple M4, PyTorch MPS backend). `environment.yml` is an equivalent conda spec.

Run everything from the project root with `PYTHONPATH=.` so that `src`, `A`, `C` and `scripts`
are importable. `src/utils/utils.get_device()` selects CUDA, then Apple MPS, then CPU.

### Docker

```bash
docker build -t otc-repro .
docker run --rm -v "$PWD/data:/work/data" -v "$PWD/results:/work/results" otc-repro
```

The image is CPU-only and runs the test suite by default. Pass any command from the sections
below to run it inside the container instead, for example
`docker run --rm -v "$PWD/data:/work/data" otc-repro python C/run_matrix.py --npz data/processed/ustc_10class.npz --classes 10`.

## Datasets

Raw captures are not redistributed. Each is available from its original source:

| Dataset | Source | Build script | Output |
|---|---|---|---|
| USTC-TFC2016 | Wang et al., ICOIN 2017; GitHub mirror `yungshenglu/USTC-TFC2016` (URL in `config/config.yaml`) | `scripts/01_download_pcap_data.py`, then `src/make_traffic_dataset.py --variant all` | `ustc_10class.npz`, `ustc_20class.npz` |
| ISCXVPN2016 | Canadian Institute for Cybersecurity, <https://www.unb.ca/cic/datasets/vpn.html> ("01" split) | `scripts/extract_iscxvpn_dataset.py` | `iscxvpn_5class.npz` |
| 5G Traffic Capture Dataset | Zenodo, doi:10.5281/zenodo.15064130 (`inference_data.zip`) | `scripts/extract_5g_dataset.py` | `g5_3class.npz` |
| IoT-23 (9-scenario subset) | Stratosphere Lab, doi:10.5281/zenodo.4743746 | `scripts/download_iot23_subset.sh`, then `scripts/extract_iot23_dataset.py` | `iot23_9class.npz` |

All outputs land in `data/processed/`. Flows are keyed by 5-tuple, the first 784 bytes form a
28x28 grayscale image, and the first 54 header bytes are zeroed. The 5G and IoT-23 builds apply a
60 s flow idle timeout, because their captures contain long single-session flows. Each
extraction script's docstring gives the expected raw-data layout. `scripts/verify_dataset.py`
checks a built archive.

## Reproducing each result

Table and figure numbers refer to the compiled paper.

### Track B: corrected release (Table IV)

```bash
PYTHONPATH=. .venv/bin/python src/cache_features.py --npz data/processed/ustc_10class.npz
HF_HUB_DISABLE_XET=1 PYTHONPATH=.:src .venv/bin/python scripts/run_b_matrix.py
HF_HUB_DISABLE_XET=1 PYTHONPATH=.:src .venv/bin/python scripts/run_b_seeds.py --only V1 V2 V3 V4 V5 Full
PYTHONPATH=. .venv/bin/python scripts/stats_b_significance.py
PYTHONPATH=. .venv/bin/python scripts/plot_b_variance.py
```

Outputs: `B_<variant>_seeds.csv`, `B_matrix.csv`, `B_significance.csv` (Wilcoxon with
Bonferroni correction, the test's p-value floor at the seed count used, paired t-test, and
Cohen's d_z), and `results/figures/b_multiseed_*.png`. Runners skip seeds that are already
recorded; pass `--force` to re-run them.

### Tracks A and A-Full (Table III)

```bash
PYTHONPATH=. .venv/bin/python A/cache_features.py --npz data/processed/ustc_20class.npz
PYTHONPATH=. .venv/bin/python A/train.py --npz data/processed/ustc_20class.npz --classes 20

PYTHONPATH=. .venv/bin/python A/cache_features_full.py --npz data/processed/ustc_20class.npz
PYTHONPATH=. .venv/bin/python A/train_full.py --npz data/processed/ustc_20class.npz --classes 20
```

Both default to 200 epochs over seeds 62, 63, 64. `--broken-conv1` reinstalls the release's
untrained frozen visual stem, for the ablation; pass it to the cache builder and the trainer
together. `bash scripts/run_a_matrix.sh` runs the whole matrix.

### Track C: baselines (Table V)

```bash
PYTHONPATH=. .venv/bin/python C/run_matrix.py --npz data/processed/ustc_10class.npz --classes 10
PYTHONPATH=. .venv/bin/python C/run_matrix.py --npz data/processed/ustc_20class.npz --classes 20
```

All four models, five seeds each. For FlowPic and FS-Net on their own published input
representations (packet-size/inter-arrival histogram, signed packet-length sequence), extract
those from the raw USTC captures first, then run with `--repr original`:

```bash
PYTHONPATH=. .venv/bin/python scripts/extract_original_representations.py
PYTHONPATH=. .venv/bin/python C/run_matrix.py --classes 10 --repr original
PYTHONPATH=. .venv/bin/python C/run_matrix.py --classes 20 --repr original
```

### Cross-dataset generalization (Table VI)

After building the three extra archives:

```bash
bash scripts/run_new_datasets_pipeline.sh
```

This runs cache builds, Track A, Track A-Full, and Track C for the 5G and IoT-23 archives. The
ISCXVPN2016 archive runs through the same commands with
`--npz data/processed/iscxvpn_5class.npz --classes 5`, passing
`--out results/comparison/A_5class.csv` (or `A_Full_5class.csv`) to the two A trainers.

### Residual-error analysis (Section "Anatomy of a Confusable Pair")

```bash
PYTHONPATH=. .venv/bin/python scripts/analyze_confusable_pair.py
PYTHONPATH=. .venv/bin/python scripts/plot_confusable_pair.py
PYTHONPATH=. .venv/bin/python scripts/test_confusable_pair_extra_features.py
```

The first reuses the A-Full 20-class checkpoints to write the per-class accuracy, confusion
matrix, and per-class physics statistics. The third needs the raw Virut and Neris captures.

### Master table

```bash
PYTHONPATH=. .venv/bin/python C/make_tables.py
```

Reads every CSV in `results/comparison/` and writes `master_table.csv`, the source for every
number in Tables III to VI.

### Architecture diagram (Fig. 1)

```bash
.venv/bin/python scripts/plot_architecture.py
```

## Building the paper

```bash
cd paper && tectonic OptimizedTrafficCLIP_IEEE.tex
```

Any LaTeX distribution with IEEEtran works as well (`pdflatex`, run twice).

## Tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest
PYTHONPATH=. .venv/bin/python -m pytest A/tests C/tests
```

`tests/` covers the shared data pipeline (flow keying, timeouts, subset derivation, cache
equivalence, the alpha fix). `A/tests/` covers the A model, conv1 adaptation, and the hypothesis
cache equivalence that gates every A-Full number. `C/tests/` covers split identity across tracks
and alignment of the original representations; it needs `data/processed/ustc_10class.npz`.

## Compute

All reported runs used one Apple M4 GPU (PyTorch MPS). Approximate wall time per seed, from the
`seconds` column of the result CSVs: Track B 4 to 35 minutes per configuration; Track A-Full
about 25 minutes at 20 classes with the prompt cache; Track C from about 1 minute (FlowPic) to
about 45 minutes (Transformer) at 20 classes.
