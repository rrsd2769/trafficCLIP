# C: Independent baselines (FlowPic, FS-Net, 1D-CNN, Transformer)

Four published-or-standard traffic-classification baselines, FlowPic (Shapira
& Shavitt), FS-Net (Liu et al.), a 1D-CNN, and a small Transformer, implemented
from scratch and trained on the existing USTC-TFC2016 data (plus three further
datasets; see `paper/review2_findings.md` Section 6). **None of these models,
nor any comparison baseline of any kind, exists anywhere in the original
work.** The inherited repository (`src/`, referred to throughout as B) is a
single pipeline for one model, `OptimizedTrafficCLIP`, with no code path that
trains, evaluates or even imports a second architecture. The review asks for A
against B against C, and C is the part with no antecedent at all rather than a
fix to something already there.

## What the original repo contains, against what C adds

| | Original repo (`src/`) | This project's addition (`C/`) |
|---|---|---|
| Models implemented | 1 (`OptimizedTrafficCLIP`, two variants of one architecture) | 4 more, from unrelated published papers or standard architectures (FlowPic: CNN; FS-Net: GRU encoder-decoder; 1D-CNN; small Transformer) |
| Comparison baselines | None; no other traffic classifier appears in the codebase, notebook or manuscript | `C/baselines/flowpic.py`, `C/baselines/fsnet.py`, `C/baselines/cnn1d.py`, `C/baselines/transformer.py`, all new files |
| Training/eval harness for a baseline | N/A, nothing to run | `C/run_matrix.py`, a new resumable multi-seed driver |
| Split-identity guarantee across models | N/A | `C/tests/test_split_identity.py`, which proves C's test set is the same set of flows A and B are scored on |
| Lines of inherited code reused by C | | **Zero.** C imports `A.dataset.stratified_splits` and `A.cache_features.{get_device,set_seed}`, both this project's own code, not `src/dataset.py` or `src/train.py` |

Taken alone, the inherited deliverable has no way to answer whether
`OptimizedTrafficCLIP` is actually better than anything else at this task, since
there is nothing in it to compare against. C exists to answer that, and
everything under `C/` is new code rather than a repaired or extended version of
existing code.

## Architecture

| Model | File | Reference | What differs from the published version |
|---|---|---|---|
| FlowPic | `C/baselines/flowpic.py:FlowPicCNN` | Shapira & Shavitt, *FlowPic* | Input is the existing 28x28 flow-byte image by default (`--repr byteimage`); `--repr original` switches to FlowPic's own packet-size/inter-arrival-time 2D histogram, extracted by `scripts/extract_original_representations.py` |
| FS-Net | `C/baselines/fsnet.py:FSNet` | Liu et al., *FS-Net* | Input is the flow image read as a 28-step row sequence by default; `--repr original` switches to a real per-packet length sequence signed by direction |
| 1D-CNN | `C/baselines/cnn1d.py` | Standard 1D convolutional stack | Reads the 784-byte flow sequence directly; no published-representation variant, since it has no "own" input format to substitute for |
| Transformer | `C/baselines/transformer.py` | Standard Transformer encoder | Reads the 784-byte sequence chunked into 28 tokens (matching FS-Net's row-grouping); self-attention cost makes it markedly slower to train (~2560s/seed at 20 classes against 1D-CNN's ~50s/seed) |

All four are trained and evaluated by `C/run_matrix.py` on the same
`data/processed/ustc_10class.npz` and `ustc_20class.npz` that A and B use (plus
the ISCXVPN2016/5G/IoT-23 archives via `--npz`/`--classes`), via the identical
70/15/15 stratified split (`A.dataset.stratified_splits`, seed 62), so C's
numbers, A's numbers and B's numbers are all measured against the same
held-out flows.

## Scope at Review 2, and what Review 3 closed

At Review 2, with two days out, C dropped the full-scope design (CIC-IDS2017
and ISCX-VPN acquisition, ET-BERT, a unified 20+-class benchmark, PCAP-level
feature extraction) in favour of a working, honest comparison on the data
already in hand: both FlowPic and FS-Net trained on the existing 28x28
flow-byte image rather than their own published input representations, and no
second dataset existed at all.

For Review 3, two of those three items were closed:

- **Each model's own published input representation.** Raw USTC-TFC2016 PCAPs
  were re-acquired and `scripts/extract_original_representations.py` builds
  FlowPic's packet-size/inter-arrival-time histogram and FS-Net's signed
  per-packet-length sequence directly from them, gated by an exact
  flow-alignment test (`C/tests/test_original_repr_alignment.py`). See
  "Results: own published representation" below.
- **Two more independent baselines**, `C/baselines/cnn1d.py` (1D-CNN) and
  `C/baselines/transformer.py` (small Transformer), on the same byte-image
  data, wired into `C/run_matrix.py` via `--repr {byteimage,original}`.
- **Three additional datasets**, ISCXVPN2016, a 5G traffic dataset, and IoT-23,
  documented in `paper/review2_findings.md` Section 6 rather than here, since
  they exercise Tracks A and A-Full as much as Track C.

**ET-BERT is still not included**, and remains the highest-compute-risk item if
a unified cross-dataset benchmark is pursued further.

Both `flowpic.py` and `fsnet.py` document the byte-image substitution in their
module docstrings, so the approximation is visible at the point of use and not
only here.

## Usage

```bash
export PYTHONPATH=.

# FlowPic + FS-Net, 5 seeds each, on the 10-class subset (comparable to B):
.venv/bin/python C/run_matrix.py --npz data/processed/ustc_10class.npz --classes 10

# Same, on the full 20-class set (comparable to A's 20-class run):
.venv/bin/python C/run_matrix.py --npz data/processed/ustc_20class.npz --classes 20

# One model only, or a rerun:
.venv/bin/python C/run_matrix.py --npz data/processed/ustc_10class.npz --classes 10 --models flowpic --force

# Split-identity check (C's test set == B's test set for the same .npz/seed):
.venv/bin/python -m pytest C/tests/ -v
```

Writes `results/comparison/C_<model>_<Nclass>.csv`, one row per seed, and
accumulates `results/comparison/C_matrix.csv` with the mean and std per model
and dataset. Resumable the same way `scripts/run_b_matrix.py` is: a
`(model, dataset)` pair already on disk is skipped unless `--force`.

## Results: byte-image representation (M4, MPS, 5 seeds x 30 epochs each)

| Model | Dataset | Accuracy | Macro F1 | CSV |
|---|---|---|---|---|
| FlowPic | 10-class | 99.39% ± 0.06pp | 99.39% ± 0.06pp | `results/comparison/C_flowpic_10class.csv` |
| FS-Net | 10-class | 99.45% ± 0.18pp | 99.45% ± 0.18pp | `results/comparison/C_fsnet_10class.csv` |
| 1D-CNN | 10-class | 99.63% ± 0.10pp | 99.63% ± 0.10pp | `results/comparison/C_cnn1d_10class.csv` |
| Transformer | 10-class | 99.50% ± 0.14pp | 99.50% ± 0.14pp | `results/comparison/C_transformer_10class.csv` |
| FlowPic | 20-class | 93.87% ± 0.35pp | 93.82% ± 0.34pp | `results/comparison/C_flowpic_20class.csv` |
| FS-Net | 20-class | 95.50% ± 0.29pp | 95.44% ± 0.28pp | `results/comparison/C_fsnet_20class.csv` |
| 1D-CNN | 20-class | 96.21% ± 0.22pp | 96.15% ± 0.23pp | `results/comparison/C_cnn1d_20class.csv` |
| Transformer | 20-class | 95.17% ± 0.41pp | 95.07% ± 0.43pp | `results/comparison/C_transformer_20class.csv` |

Mean and std summary: `results/comparison/C_matrix.csv`.

**All four simple baselines beat B's honestly-evaluated Full variant by a wide
margin.** B's best honest number, Full with dynamic and stats prompting and the
leaking eval path fixed, is 88.13% ± 2.12pp accuracy and 88.29% ± 2.15pp F1
(5-seed mean) on the identical 10-class test partition
(`results/comparison/B_Full_seeds.csv`), while all four Track C baselines clear
99.3% on the same held-out flows. They also land close to A's reimplementation
of the base paper (99.75% on 10-class, 96.64% on 20-class) despite being far
smaller, simpler models trained from scratch with no pretrained backbone at
all.

This does not make the four baselines better traffic classifiers in any general
sense. All four read the flow-byte image directly, which is an easy
discriminative signal for this particular 10/20-class USTC-TFC2016 split, and a
plain CNN, GRU, or Transformer can fit it with no architectural bottleneck in
the way. B's OptimizedTrafficCLIP by contrast routes everything through two
*frozen* pretrained backbones, ResNet-50 and BERT, and trains only small
projection and fusion heads downstream of them, which is a much narrower path
to the same information, on top of the six documented defects. The gap between
B's 88.1% and C's 99.4%+ is a property of B's architecture and training budget
on this task, not evidence that the classification problem is hard: A and C
both show it is not, once nothing throttles the signal reaching the
classifier.

## Results: own published representation (M4, MPS, 5 seeds x 30 epochs each)

| Model | Dataset | Accuracy | Macro F1 | CSV |
|---|---|---|---|---|
| FlowPic | 10-class | 80.71% ± 0.06pp | 80.82% ± 0.07pp | `results/comparison/C_flowpic_10class_orig.csv` |
| FS-Net | 10-class | 91.67% ± 0.39pp | 90.97% ± 0.39pp | `results/comparison/C_fsnet_10class_orig.csv` |
| FlowPic | 20-class | 64.63% ± 0.57pp | 65.07% ± 0.41pp | `results/comparison/C_flowpic_20class_orig.csv` |
| FS-Net | 20-class | 80.96% ± 0.95pp | 79.86% ± 0.67pp | `results/comparison/C_fsnet_20class_orig.csv` |

```bash
# Own published representation (requires data/raw/<class>/*.pcap; see
# scripts/extract_original_representations.py):
.venv/bin/python C/run_matrix.py --classes 10 --repr original --models flowpic fsnet
```

**FlowPic's own representation is measurably worse than the byte-image
substitution; FS-Net's own representation is closer but still worse.** FlowPic
loses 18.7pp accuracy at 10 classes and 29.2pp at 20; FS-Net loses 7.8pp and
14.5pp respectively. This confirms, quantitatively, the direction this
document's own substitution scope note always predicted, and closes that
caveat rather than leaving it as an untested assumption. Neither FlowPic's nor
FS-Net's original paper evaluates on USTC-TFC2016, so there is still no
same-dataset published number for either representation to compare against;
the comparison above is within-dataset, cross-representation, not
published-versus-reproduced.
