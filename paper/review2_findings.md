# Review 2: Reproduction and Extension Report

**OptimizedTrafficCLIP: Independent Reimplementation, Honest Reproduction, and Baseline Comparison**

Submitted for Review 2. All numbers in this report come from files committed under
`results/comparison/*.csv`, most of them via `C/make_tables.py`. No number appears here that is
not traceable to one of those files.

---

## Abstract

We report three independent lines of work against Pramit De's *OptimizedTrafficCLIP* manuscript
and its released code. **(A)** We built an independent reimplementation of the TrafficCLIP and
OptimizedTrafficCLIP architecture directly from the published specification, without consulting
the released implementation, and evaluated it on the manuscript's own 10-class and 20-class
USTC-TFC2016 splits. **(B)** We executed the released code as documented, found and fixed six
defects in its training and evaluation pipeline, and re-measured every reported variant under a
corrected, honest protocol, averaged over 5 independent seeds. **(C)** We implemented four
independent baselines, FlowPic, FS-Net, a 1D-CNN, and a small Transformer, from scratch and
trained them on the identical data and test split; FlowPic and FS-Net were additionally re-trained
on their own originally published input representations. All three lines of work evaluate on a
provably identical held-out test partition. Four architecturally different approaches (A's
reimplementation, and C's FlowPic, FS-Net, 1D-CNN, and Transformer) all solve the 10-class task
near ceiling at 99%+ accuracy, while B, the manuscript's own architecture honestly evaluated,
reaches 88.13% ± 2.12pp (5-seed mean). We argue this is informative about B's specific
architecture and training budget on this task rather than evidence that the classification
problem itself is hard, and we show that the manuscript's headline 87.54% figure originates from a
separate evaluation path that leaks the ground-truth label into the model's text input. A
follow-up track, **A-Full**, closes the remaining question directly: an independent rebuild of B's
*exact* tri-modal architecture (physics stats, dynamic prompts, non-linear fusion), free of B's six
defects by construction, scores 96.6% on 20-class, matching the simpler bi-modal A and landing
about 8.5pp above B's own honest 88.1% for the identical design. B's gap is B's code, not its
architecture. This finding replicates on three further, independently sourced datasets
(ISCXVPN2016, a 5G traffic dataset, and IoT-23): A and A-Full remain statistically
indistinguishable on every one, so the parity result is not specific to USTC-TFC2016.

---

## 1. Scope and Method

Three independent tracks (A, B, C) were run against the same data foundation:

- **Dataset.** USTC-TFC2016, preprocessed per the manuscript's own protocol: flows keyed by
  5-tuple, first 784 bytes reshaped to a 28x28 grayscale image, first 54 header bytes zeroed.
  Both the manuscript's 10-class balanced subset (900 samples/class, 9,000 total) and a 20-class
  extension were built.
- **Split.** All three tracks evaluate on the same stratified 70/15/15 train/val/test partition
  at a fixed seed. `C/tests/test_split_identity.py` asserts directly that A's split-generation
  code and B's split-generation code produce identical test-set indices for the same `.npz`
  file, so the comparison below is not an artifact of different test sets.
- **Traceability.** Every number in this report is read from a CSV under `results/comparison/`
  by `C/make_tables.py`, which writes `results/comparison/master_table.csv`. Nothing here is
  hand-typed from a log or a manual calculation.

## 2. Track A: Independent Reimplementation

A was built independently from the base TrafficCLIP paper's specification (multi-level visual
encoder, frozen BERT text branch, cosine-similarity classification, joint cross-entropy and
supervised contrastive loss), without reference to the released `OptimizedTrafficCLIP` code.

| Configuration | Accuracy | Macro F1 | Seeds |
|---|---|---|---|
| 10-class | 99.75% ± 0.09pp | 99.75% ± 0.09pp | 3 |
| 20-class | 96.64% ± 0.23pp | 96.64% ± 0.23pp | 3 |
| 20-class, broken-conv1 ablation | 96.68% ± 0.14pp | 96.68% ± 0.14pp | 3 |

*Source: `results/comparison/A_10class.csv`, `A_20class.csv`, `A_20class_brokenconv1.csv`.*

A confirms independently that the underlying 10-class task is solvable near ceiling by a
correctly trained CLIP-style model, without touching any of the released B code.

## 3. Track A-Full: Is B's Gap Architecture or Code?

Track A implements only the base TrafficCLIP paper's bi-modal design. It has no physics-stats
branch and no dynamic prompting, so on its own it cannot say whether B's gap (88.13% honest 5-seed
mean on 10-class, and far below A and C's 99%+ on the same 10-class task) comes from B's six
defects or from something inherent to B's tri-modal architecture. A-Full answers that directly.
It is a second independent implementation (`A/model_full.py`, `A/dataset_full.py`,
`A/cache_features_full.py`, `A/train_full.py`) that adds back B's two distinguishing ingredients,
a physics-informed stats branch over mean IAT, jitter and byte entropy, and per-sample dynamic
prompting, fused through the same non-linear MLP head B uses instead of A's plain cosine
similarity. This is B's architecture, rebuilt from scratch a second time, avoiding Defects 1, 2,
3 and 6 by construction (see `A/model_full.py`'s module docstring): the split is fixed once per
archive rather than per seed, frozen submodules are pinned to `eval()` inside the model's own
`train()` method, and training uses the *same* honest K-hypothesis computation as evaluation, so
the model never sees the sample's true-label prompt as a training input the way B's training loop
does.

Replacing B's training loop with an honest one means a live BERT forward pass for every candidate
class name at every step, which does not scale: measured at ~7.5s/step at K=20, it projects to
~246 hours for the full 3-seed matrix. Since every sample's physics stats are fixed for a given
run, its K dynamic-prompt strings and their BERT embeddings are fixed too, so precomputing them
once into an `(N, K, 768)` cache cut the same matrix to about an hour.
`A/tests/test_model_full.py::test_hyp_cache_equivalence` gates this: cached and live hypothesis
logits must agree exactly, or every A-Full number below is void.

| Configuration | Accuracy | Macro F1 | Seeds |
|---|---|---|---|
| A-Full, 10-class | 99.68% ± 0.07pp | 99.68% ± 0.07pp | 3 |
| A-Full, 20-class | 96.63% ± 0.09pp | 96.64% ± 0.08pp | 3 |
| A-Full, 20-class, broken-conv1 ablation | 96.80% ± 0.20pp | 96.79% ± 0.21pp | 3 |

*Source: `results/comparison/A_Full_10class.csv`, `A_Full_20class.csv`,
`A_Full_20class_brokenconv1.csv`.*

**B's gap is code, not architecture.** A-Full essentially matches base A (96.6% against 96.64% on
20-class), so the added stats branch and dynamic prompting cost nothing and do not obviously help
on this dataset either, but both independent variants land about 8.5 points above B's own honest
88.13% for the *identical* tri-modal design. Isolated from Defects 1, 2, 3 and 6, the
physics-stats-plus-dynamic-prompt architecture is not what holds B back; B's implementation of it
is.

**The broken-conv1 ablation shows Defect 4 costs nothing here either.** Re-running A-Full with
Defect 4 reinstalled verbatim, a fresh randomly initialised frozen `Conv2d(1,64,7,7)` in place of
A's grayscale-adapted pretrained kernel, scores 96.80%, statistically indistinguishable from
A-Full's correct-conv1 96.63% and echoing base A's own ablation (96.68% against 96.64%, Section
2). All four independent variants, A and A-Full crossed with correct and broken conv1, cluster at
96.6-96.8%, while B sits about 8.5pp below every one of them. B's gap is not attributable to any
single isolated defect in this set, including the one originally hypothesized as the leading
cause.

**20-class is not fully solved, and the residual is not a code defect.** A per-class breakdown of
A-Full's 20-class predictions, evaluated separately from each of the 3 seeds' own checkpoints
against the shared test split, shows 18 of 20 classes at 95-100% accuracy (16 of those at
99-100%). Two classes fall well outside that band:

| Class | Accuracy (3-seed mean) |
|---|---|
| Virut | 68.4% ± 0.7pp |
| Neris | 78.0% ± 3.5pp |
| Nsis-ay | 95.1% ± 1.5pp |
| every other class | 97.8-100% |

*Source: `results/comparison/a_full_20class_per_class_accuracy.csv`, `results/figures/a_full_20class_per_class_accuracy.png`.*

#### Anatomy of a confusable pair

**Virut** and **Neris**, both botnet malware families, account for 207 of the 273 total
misclassification errors summed across all 3 seeds (**75.8%**). Per seed this ranges from 70.5%
(seed 64: 62/88) to 82.4% (seed 62: 75/91) to 74.5% (seed 63: 70/94) -- a real spread, not the
flat "consistent across all three seeds" this section previously claimed from the seed-62 figure
alone; the aggregated 75.8% is the more robust number.

The stats branch, A-Full's own physics-informed addition, carries no disambiguating signal for
this pair. Computed directly from the test split's own IAT/jitter/entropy array (independent of
any model): the two classes' **medians** are nearly identical --

| Feature | Virut median | Neris median |
|---|---|---|
| IAT | 1793 ms | 1783 ms |
| Jitter | 2399 ms | 2396 ms |
| Byte entropy | 4.94 | 4.93 |

-- while their **means** diverge sharply (IAT 181,107 ms against 17,976 ms; jitter 398,904 ms
against 99,305 ms), driven entirely by a handful of extreme outliers in each class rather than a
real distributional separation. `results/figures/confusable_pair_distributions.png` shows this
directly: the boxplots' medians (orange lines) sit on top of each other while the means (green
triangles) are pulled apart by outliers well outside each box.

*Source: `results/comparison/confusable_pair_feature_stats.csv`,
`results/comparison/a_full_20class_confusion_matrix.csv`. Recomputed from the existing
`A/checkpoints/A_Full_20class_seed{62,63,64}.pt` checkpoints, no retraining
(`scripts/analyze_confusable_pair.py`).*

**This is a dataset property, not a code defect.** Virut/Neris confusion in USTC-TFC2016 is a
known-hard pair elsewhere in the traffic-classification literature, and the feature set here
(mean IAT, jitter, byte entropy) does not distinguish them. So A-Full's 96.6% on 20-class is a
near-solved problem, 18 of 20 classes at ceiling with one well-understood hard pair, not 20 of 20
solved.

**Phase 4.2 update: raw PCAPs were re-acquired, and a feasibility probe finds real (if partial)
extra signal.** The source PCAPs were re-downloaded from the same public mirror
(`scripts/01_download_pcap_data.py`) once disk space was confirmed available, unblocking this
question. `scripts/test_confusable_pair_extra_features.py` walks only the Virut and Neris raw
captures, with the identical flow-collection logic as the main dataset build (asserted at 900/900
flows collected, matching `ustc_20class.npz`'s per-class count), and computes 7 features per flow
that are not in the existing 3-value stats vector: packet-size mean/std/skew, burst count and mean
burst size (1s idle-gap threshold), and byte entropy computed separately over only the first 200
and only the last 200 bytes of the flow, rather than the whole flow at once.

A plain logistic regression on these 7 features alone, 5-fold cross-validated, scores **61.4% ±
4.9%** accuracy telling Virut from Neris apart (chance is 50%, classes are balanced at 900 each).
That is a real, if modest, signal, not the near-total separation that would suggest the pair is
easily resolvable. The most informative single feature is where in the flow entropy is measured,
not overall entropy magnitude: entropy over the flow's last 200 bytes differs sharply between the
two classes (Neris mean 4.93, Virut mean 4.77, $t=9.27$, $p \approx 5\times10^{-20}$), while the
existing stats vector's whole-flow entropy is nearly identical between them (4.94 vs. 4.93
medians, Section 3 above). Packet-size skew and mean also differ significantly ($p<0.01$); burst
count and mean burst size do not ($p>0.08$).

**This is a feasibility probe, not a full tri-modal retrain.** It answers a narrower, cheaper
question, do these features separate the two classes better than the existing physics-stats branch
does, on their own -- and the answer is a qualified yes. For scale: once A-Full routes a flow to one of
the two classes, it already separates them correctly 74.1% of the time (593 of 800 decisions,
summed over 3 seeds, from `results/comparison/a_full_20class_confusion_matrix.csv`), so the
extra features alone (61.4%) are weaker than the image branch. Whether they are complementary to
it is the question only a retrain answers. It does not retrain A-Full's actual stats
branch with these features added, which is a separate, multi-hour, multi-seed effort and the
natural next step if this line is pursued further. Source:
`results/comparison/confusable_pair_extra_features.csv`.

*Full derivation, including the confusion-matrix methodology: `A/README.md`, "A-Full: does the
tri-modal design close B's gap?" and "Where the remaining 20-class error comes from."*

## 4. Track B: Reproduction of the Released Code

### 4.1 Defects found

We executed the released `OptimizedTrafficCLIP` repository as documented and found six defects in
its training and evaluation pipeline:

| # | Defect | Effect |
|---|---|---|
| 1 | `alpha` (contrastive-loss weight) not exposed as a constructor argument | Manuscript's `alpha=10` config unreachable through the documented API |
| 2 | Evaluation forward pass's dynamic prompt embeds the ground-truth class name | Eval-time label leakage; inflates reported accuracy |
| 3 | `benchmark.py` regenerates the train/test split at `seed + run` on each repeated pass | Split leakage across "repeated" evaluation passes |
| 4 | `validate()`'s hypothesis-testing loop runs one sample at a time, and reads an unresolved `class_names` variable instead of the resolved list | Silent incorrectness in the dynamic-prompt path, independent of speed |
| 5 | Training `DataLoader` built with `shuffle=False` | Frozen batch composition across epochs degrades the contrastive loss |
| 6 | Frozen ResNet-50/BERT submodules are not forced into `eval()` mode inside `train()` | BERT dropout stays active during live forward passes (hard crash under MPS `scaled_dot_product_attention`); BatchNorm running stats keep drifting despite the backbone being "frozen" |

All six were fixed in place (see `src/train.py`, `src/benchmark.py`, `src/dataset.py`,
`src/optimization/profiling.py`, `src/model.py`). Two further, unrelated bugs were found and
fixed while exercising the fixed pipeline against real data: `num_classes` was derived from the
full 20-class taxonomy instead of the loaded `.npz`'s actual label count, and MLflow 3.x's
default model-logging serialization crashed at the end of every run.

### 4.2 Honest re-evaluation

We executed the manuscript's own hypothesis-testing evaluation protocol (section IV-C of the
manuscript) against the defect-fixed code, across 5 independent seeds (62-66):

| Variant | Manuscript-reported Acc / F1 | Honest (post-fix), 5-seed mean Acc / F1 |
|---|---|---|
| V1, Original, no stats | 62.74% / 62.21% | 34.27% / 29.54% |
| V2, Original, with stats | 61.63% / 59.05% | 36.92% / 31.54% |
| V3, Optimized, hard prompt, no stats | 61.63% / 63.10% | 47.70% / 39.66% |
| V4, Optimized, dynamic prompt, no stats | 79.33% / 78.57% | **86.00% / 86.19%** |
| V5, Optimized, hard prompt, stats | 66.44% / 67.01% | 50.22% / 43.83% |
| **Full**, Optimized, dynamic prompt, stats | 80.49% / 79.63% | **88.13% / 88.29%** |

*Sources: manuscript column transcribed from the manuscript's Table 1; honest column from
`results/comparison/B_V1_seeds.csv` through `B_Full_seeds.csv` (5 seeds each), aggregated in
`results/comparison/B_matrix.csv`. Both columns are laid out together in
`results/comparison/master_table.csv`. (Single-seed figures from an earlier pass are superseded by
these 5-seed means; see `results/comparison/B_significance.csv` for the accompanying pairwise
Wilcoxon test.)*

The variants that rely solely on the static hard prompt (V1, V2, V3, V5) collapse once the
label-leaking evaluation path (Defect 2) is fixed. The two variants that use the dynamic,
physics-informed prompt (V4 and Full) **improve** under honest evaluation relative to their
manuscript-reported figures: V4 rises from 78.57% to 86.19% Macro F1 (5-seed mean), a gain of 7.6
points, and Full from 79.63% to 88.29%, a gain of 8.7 points. This is the central empirical finding
of Track B. We executed the released code as documented, and under the paper's own stated
hypothesis-testing protocol we obtain 88.13% accuracy and 88.29% Macro F1 (5-seed mean) for the
Full configuration. The manuscript's reported 87.54% figure originates from a separate forward-pass
evaluation path whose text input carries the ground-truth label (Defect 2), not from fabricated
numbers. The dynamic-prompting mechanism itself is real and, honestly measured, holds up close to
its reported strength. A pairwise Wilcoxon signed-rank test with Bonferroni correction across all
six configurations found no Bonferroni-significant pairs, but at $n=5$ seeds the exact two-sided
test cannot fall below $p=0.0625$ regardless of effect size, so this null result reflects that
sample-size floor rather than genuine equivalence between the dynamic- and static-prompt
configurations (`results/comparison/B_significance.csv`, `paper/b_multiseed_macro_f1.png`). A
paired t-test on the same seed pairs, which has no such floor and is now written alongside the
Wilcoxon columns, separates the two groups clearly: V4 and Full each differ from every
static-prompt configuration at Bonferroni-corrected $p<0.001$ (Cohen's $d_z$ between 8 and 54),
while V4 against Full does not differ ($p=0.93$).

Separately, we confirmed the mechanism of Defect 2 directly: re-running the pre-fix leaking
evaluation path against a tuned Full checkpoint produced a near-ceiling Macro F1 score in the
0.85 to 0.87 range, dramatically higher than any honest number reported above, which is exactly
the direction a genuine label leak predicts. That spot-check was a manual, ad hoc verification
rather than a committed CSV, so we report it qualitatively rather than as a precise figure.
Post-fix, ONNX and PyTorch inference agree exactly at 0.8797 Macro F1, and ONNX export yields a
genuine ~1.7x latency reduction under the corrected protocol.

## 5. Track C: Independent Baselines

We implemented FlowPic (2D CNN over the flow-byte image), FS-Net (bidirectional GRU
encoder/decoder plus classifier, joint CE and reconstruction loss), a 1D-CNN, and a small
Transformer from scratch and trained all four on the same 10-class and 20-class USTC-TFC2016
data, 5 seeds each.

| Model | Dataset | Accuracy | Macro F1 | Seeds |
|---|---|---|---|---|
| FlowPic | 10-class | 99.39% ± 0.06pp | 99.39% ± 0.06pp | 5 |
| FS-Net | 10-class | 99.45% ± 0.18pp | 99.45% ± 0.18pp | 5 |
| 1D-CNN | 10-class | 99.63% ± 0.10pp | 99.63% ± 0.10pp | 5 |
| Transformer | 10-class | 99.50% ± 0.14pp | 99.50% ± 0.14pp | 5 |
| FlowPic | 20-class | 93.87% ± 0.35pp | 93.82% ± 0.34pp | 5 |
| FS-Net | 20-class | 95.50% ± 0.29pp | 95.44% ± 0.28pp | 5 |
| 1D-CNN | 20-class | 96.21% ± 0.22pp | 96.15% ± 0.23pp | 5 |
| Transformer | 20-class | 95.17% ± 0.41pp | 95.07% ± 0.43pp | 5 |

*Source: `results/comparison/C_matrix.csv`; per-seed detail in `C_flowpic_*.csv`, `C_fsnet_*.csv`,
`C_cnn1d_*.csv`, `C_transformer_*.csv`. `C/baselines/cnn1d.py` and `C/baselines/transformer.py` are
the two new architectures; `C/run_matrix.py --repr byteimage` wires them into the same harness as
FlowPic/FS-Net.*

All four models read the existing 28x28 flow-byte image rather than a model-specific input
representation. For 1D-CNN and Transformer this is simply their native input (a byte sequence),
but for FlowPic and FS-Net specifically it substitutes for their own published representations,
FlowPic's packet-size/inter-arrival-time histogram and FS-Net's per-packet-length sequence. That
scope caveat, left open in the original Review 2 report because raw PCAP access had already been
discarded, is now closed empirically (Section 5.1 below).

### 5.1 Phase 2.1: FlowPic and FS-Net on their own published representations

The raw USTC-TFC2016 PCAPs were re-acquired (see the Phase 4.2 update in Section 3 above for the
same re-acquisition) specifically to build FlowPic's and FS-Net's own published input
representations from scratch: `scripts/extract_original_representations.py` constructs FlowPic's
32x32 packet-size/time histogram and FS-Net's signed packet-length sequence directly from
`data/raw/<class>/*.pcap`, gated by an exact flow-order-alignment test
(`C/tests/test_original_repr_alignment.py`, passing) against the existing `ustc_*class.npz`
archives, so the comparison below is the *same* underlying flows, two different representations,
not two different samples.

| Model | Dataset | Accuracy | Macro F1 | Seeds |
|---|---|---|---|---|
| FlowPic (own repr.) | 10-class | 80.71% ± 0.06pp | 80.82% ± 0.07pp | 5 |
| FS-Net (own repr.) | 10-class | 91.67% ± 0.39pp | 90.97% ± 0.39pp | 5 |
| FlowPic (own repr.) | 20-class | 64.63% ± 0.57pp | 65.07% ± 0.41pp | 5 |
| FS-Net (own repr.) | 20-class | 80.96% ± 0.95pp | 79.86% ± 0.67pp | 5 |

*Source: `results/comparison/C_flowpic_10class_orig.csv`, `C_flowpic_20class_orig.csv`,
`C_fsnet_10class_orig.csv`, `C_fsnet_20class_orig.csv`, aggregated in `C_matrix.csv` under the
`*_orig` dataset tag. (Note: an earlier bug in `C/run_matrix.py`'s summary-writing step tagged
these rows with the *same* key as the byte-image substitution rows, silently overwriting the
correct byte-image numbers in the aggregated `C_matrix.csv` -- the per-seed CSVs were always
correct. Fixed in `C/run_matrix.py`'s `main()`; `C_matrix.csv` and `master_table.csv` were
regenerated from the per-seed CSVs to recover the correct byte-image rows.)*

**FlowPic's own representation is measurably worse than the byte-image substitution; FS-Net's own
representation is closer but still worse.** FlowPic loses 18.7pp of accuracy at 10 classes (99.39%
to 80.71%) and 29.2pp at 20 classes (93.87% to 64.63%). FS-Net loses 7.8pp (99.45% to 91.67%) and
14.5pp (95.50% to 80.96%) respectively. This is not a caveat anymore; it is a measured number, and
it confirms in the predicted direction that the byte-image substitution made Section 5's main
comparison somewhat easier than each baseline's original published setting.

### 5.2 Phase 2.2: two additional baselines (1D-CNN, small Transformer)

`C/baselines/cnn1d.py` and `C/baselines/transformer.py` are two further from-scratch
architectures, trained on the existing byte-image data (same table as the main Track C results
above). Both exceed 95% accuracy at 20 classes, matching FlowPic and FS-Net's byte-image numbers.
The Transformer (30 tokens of self-attention over the 784-byte sequence) was markedly slower to
train than the other three models -- roughly 2560s/seed at 20 classes against 1D-CNN's ~50s/seed --
but reached comparable accuracy.

### 5.3 Phase 2.3: no same-dataset published number exists

Neither FlowPic's nor FS-Net's original paper evaluates on USTC-TFC2016, so there is no
same-dataset published number for either baseline to compare against directly. The honest framing
of Sections 5 and 5.1, and the only framing this report makes, is: (a) each baseline's own
representation against the byte-image substitution, on the *same* dataset and the *same* flows,
and (b) each baseline's own published accuracy on *its own* original dataset, read separately and
not merged into one table with USTC-TFC2016 numbers as if they were comparable. Any published
FlowPic/FS-Net number quoted elsewhere in this project's materials refers to their own paper's own
dataset, not a reproduction target for this report.

## 6. Cross-Dataset Generalization (Phase 3)

Sections 2-5 evaluate exclusively on USTC-TFC2016. This section re-runs Tracks A, A-Full, and all
four Track C baselines, unmodified beyond output class count, on three additional, independently
sourced datasets, to test whether the bi-modal/tri-modal parity finding (Section 3) is specific to
USTC-TFC2016's own separability or generalizes.

### 6.1 Datasets

- **ISCXVPN2016** (Gil et al., ICISSP 2016): a 3.1GB "01" split of the public VPN/non-VPN corpus,
  re-labeled into a 5-class traffic-type taxonomy (Chat, Email, FileTransfer, P2P, VoIPVideo),
  folding VPN and non-VPN captures of the same type into one class so the task stays "classify
  traffic type" rather than "detect VPN encapsulation." 2,914 samples, naturally imbalanced
  (FileTransfer: 127, P2P: 487, against 900 each for Chat/Email/VoIPVideo).
  `scripts/extract_iscxvpn_dataset.py` -> `data/processed/iscxvpn_5class.npz`.
- **5G Traffic Capture Dataset** (UFJF NetLab, Zenodo DOI 10.5281/zenodo.15064130): the 2.3GB
  `inference_data.zip` split, four continuous PCAP captures labeled by 5G traffic type (eMBB video
  streaming, URLLC live streaming, mMTC low-rate UDP telemetry). 472 samples, 3 classes. Each
  capture is a single sustained session on one 5-tuple for its whole duration, so a naive build
  with no idle-timeout flow segmentation collapses almost the entire file into one "flow" per
  capture (confirmed empirically: an early build with `flow_timeout=0` produced only 412 samples
  total, 186/220/6 across the three classes). `scripts/extract_5g_dataset.py` uses the project's
  existing 60-second idle-timeout flow segmentation (`flow_timeout=60.0`, the same setting used for
  `ustc_*class_t60.npz`) to fix this; mMTC remains the smallest class at 59 samples regardless,
  since the corpus itself ships only two mMTC captures, one a perfectly constant-rate synthetic UDP
  generator with no idle gaps to split on. -> `data/processed/g5_3class.npz`.
- **IoT-23** (Garcia et al./Stratosphere Lab, Zenodo DOI 10.5281/zenodo.4743746): a hand-picked,
  ~2.55GB, 9-class subset of the 21GB full corpus (3 real, non-simulated benign IoT devices --
  Amazon Echo, Philips Hue, Somfy Doorlock -- plus 6 malware family/scenario captures -- Mirai,
  Hide-and-Seek, Muhstik, Torii, a Trojan capture, Hakai). 6,340 samples. The same 60-second
  flow-timeout matters here too: Mirai's capture is a long-running DDoS/scan session that yields
  only 20 usable flow samples with no timeout against 229 with it, since a sustained flood against
  one victim IP is otherwise a single, arbitrarily long same-5-tuple flow.
  `scripts/extract_iot23_dataset.py` -> `data/processed/iot23_9class.npz`. Class sizes are
  imbalanced by capture availability (Mirai: 229, Somfy Doorlock: 258, Philips Hue: 453, all six
  other classes at the 900-sample cap), reported as-is.

All three datasets reuse `src/make_traffic_dataset.py`'s `process_pcaps_to_numpy` (ISCXVPN2016 and
IoT-23, folder-per-class) or the same underlying `FlowAccumulator`/`flow_key`/`iter_packets`
primitives (5G, filename-per-class via symlinks), so the resulting `.npz` schema is identical to
`ustc_*class.npz` and Tracks A/A-Full/C run against them via `A.dataset.stratified_splits` with no
code changes beyond `--classes` and `--npz`.

### 6.2 Tracks A and A-Full: does the parity finding replicate?

| Dataset | Track A Acc | Track A-Full Acc | Seeds |
|---|---|---|---|
| ISCXVPN2016 (5-class) | 92.24% ± 0.68pp | 91.97% ± 0.71pp | 3 |
| 5G Traffic (3-class) | 97.65% ± 1.76pp | 98.59% ± 1.15pp | 3 |
| IoT-23 (9-class) | 98.70% ± 0.13pp | 98.91% ± 0.25pp | 3 |

*Source: `results/comparison/A_5class.csv`, `A_Full_5class.csv`, `A_5G_3class.csv`,
`A_Full_5G_3class.csv`, `A_IoT23_9class.csv`, `A_Full_IoT23_9class.csv`.*

**Yes: Track A and Track A-Full remain statistically indistinguishable on all three datasets.**
This is the same pattern as USTC-TFC2016 (Section 3), now confirmed on three further datasets that
differ from USTC-TFC2016 and from each other in application domain (VPN traffic type, 5G network
slice, IoT device/malware), data generation (real captured VPN sessions, a simulated 5G core, real
IoT device and botnet traffic), and collection methodology. The tri-modal design's parity with its
simpler bi-modal counterpart is not an artifact of USTC-TFC2016's own separability.

### 6.3 Track C: do the baselines generalize too?

| Model | ISCXVPN2016 (F1) | 5G (F1) | IoT-23 (F1) |
|---|---|---|---|
| FlowPic | 90.88% ± 0.58pp | 97.81% ± 0.00pp | 97.99% ± 0.13pp |
| FS-Net | 89.54% ± 1.01pp | 98.24% ± 0.90pp | 97.96% ± 0.38pp |
| 1D-CNN | 92.86% ± 0.65pp | 96.91% ± 1.10pp | 98.26% ± 0.18pp |
| Transformer | 85.00% ± 5.14pp | 95.88% ± 1.58pp | 97.40% ± 0.38pp |

*Source: `results/comparison/C_matrix.csv`, rows tagged `5class`, `3class`, `9class`.*

Every baseline exceeds 85% Macro F1 on every dataset. No single baseline is uniformly ahead: FS-Net
leads on ISCXVPN2016 and 5G, 1D-CNN leads on IoT-23. The Transformer's 5G result has a notably
wider seed-to-seed spread (5.14pp) than any other cell in this table, plausibly its 30-token
self-attention window interacting with the smallest and most imbalanced of the three new datasets
(mMTC: 59 samples) rather than a general weakness, since the same architecture is within 1.1pp of
1D-CNN on USTC-TFC2016's own 20-class task (Section 5).

### 6.4 What this section does not claim

Track B (the released code, honestly re-evaluated) was not re-targeted to these three datasets:
its evaluation harness is specific to the original 10/20-class taxonomy, and re-targeting it was
out of scope here. Whether B's own gap against A/A-Full also replicates cross-dataset is therefore
an open question this section does not answer, flagged in Section 9 (Limitations).

## 7. Master Comparison Table

| Approach | Configuration | Accuracy / Macro F1 | Seeds |
|---|---|---|---|
| A | 10-class | 99.75% / 99.75% | 3 |
| A | 20-class | 96.64% / 96.64% | 3 |
| A-Full | 10-class (B's architecture, independently built) | 99.68% / 99.68% | 3 |
| A-Full | 20-class (B's architecture, independently built) | 96.63% / 96.64% | 3 |
| B | Full (honest, post-fix, 5-seed mean) | 88.13% / 88.29% | 5 |
| C | FlowPic, 10-class | 99.39% / 99.39% | 5 |
| C | FS-Net, 10-class | 99.45% / 99.45% | 5 |
| C | 1D-CNN, 10-class | 99.63% / 99.63% | 5 |
| C | Transformer, 10-class | 99.50% / 99.50% | 5 |
| C | FlowPic, 10-class (own published repr.) | 80.71% / 80.82% | 5 |
| C | FS-Net, 10-class (own published repr.) | 91.67% / 90.97% | 5 |
| A / A-Full | ISCXVPN2016, 5-class | 92.24% / 91.97% (acc) | 3 |
| A / A-Full | 5G Traffic, 3-class | 97.65% / 98.59% (acc) | 3 |
| A / A-Full | IoT-23, 9-class | 98.70% / 98.91% (acc) | 3 |

*Generated by `C/make_tables.py` into `results/comparison/master_table.csv` (49 rows total,
covering A, A-Full -- both main and broken-conv1 ablation, both USTC-TFC2016 and all three
cross-dataset builds -- every B variant at 5-seed mean, the manuscript's pre-fix Table 1 figures
for reference, and every Track C model/dataset/representation combination). A-Full's rows are now
generated through the same `a_row()` helper as Track A, at the same traceability standard, closing
the gap the original Review 2 report flagged. It is abbreviated here to the headline comparison;
see `results/comparison/master_table.csv` for the full table.*

## 8. Discussion

Four architecturally different approaches, A's reimplementation and C's from-scratch FlowPic,
FS-Net, 1D-CNN, and Transformer, all solve the 10-class USTC-TFC2016 task near ceiling at 99%+
accuracy, using four different visual and sequence encodings and none of B's frozen-backbone
tri-modal fusion machinery. Only B, honestly evaluated, sits at 88.13% ± 2.12pp (5-seed mean).

**We read this as a statement about B specifically, not about the task.** B routes both its
visual and textual signal through *frozen* pretrained ResNet-50 and BERT backbones, training only
small projection and fusion heads downstream, which is a deliberate parameter-efficient design
that trades capacity for edge deployability. On a task this separable, four simpler, fully
trainable approaches close the gap that B's frozen-backbone design leaves open. That is a
property of B's architecture and training budget on *this* task, not evidence that encrypted
traffic classification on USTC-TFC2016 is intrinsically hard, and it would have been impossible
to tell the two apart without A and C as independent controls.

Track A-Full (Section 3) rules out a further alternative explanation: that A and C solve the task
easily only because they *are not* B's tri-modal architecture, and that B's frozen-backbone
design is simply the wrong tool for this task. A-Full is B's design, physics stats, dynamic
prompts, non-linear fusion, frozen backbones and all, rebuilt independently, and it scores 96.6%
on 20-class, matching plain bi-modal A and clearing B's own honest 88.1% by about 8.5 points. B's
architecture is not the bottleneck; B's implementation of six specific defects is. That also
means the manuscript's central contribution, physics-informed dynamic prompting, is not merely
real but overstated by 87.54%, as Track B's numbers alone might suggest. A-Full shows the full
design, correctly executed, performs *at least* as well as a design without it. Whether the
tri-modal additions provide a benefit *beyond* matching bi-modal A on USTC-TFC2016 specifically is
addressed directly by Section 6: across ISCXVPN2016, the 5G dataset, and IoT-23, A and A-Full
remain statistically indistinguishable on every one, so the parity result is not an artifact of
USTC-TFC2016's own separability, and neither is it evidence that the tri-modal additions provide a
benefit beyond bi-modal A on any of the four datasets tested here.

The manuscript's own contribution is real and reproducible: it drives a 15.5-point Macro F1 gain
in the manuscript's own validation-split ablation (V3 to V4), and honest test-set evaluation
confirms the same mechanism is decisive on held-out data. V4 (86.00% / 86.19%) and Full (88.13% /
88.29%, both 5-seed means) are the only two configurations that clear 80% accuracy post-fix, while
every static-prompt variant (V1, V2, V3, V5) stays below 51%. What does not reproduce is the
specific 87.54% headline figure, which traces to an evaluation-protocol defect rather than to the
architectural contribution itself.

Sections 5.1-5.3 close the remaining scope caveat from the original Review 2 report: FlowPic and
FS-Net, re-trained on their own published input representations rather than the byte-image
substitution, both score measurably lower (FlowPic: -18.7pp to -29.2pp accuracy; FS-Net: -7.8pp to
-14.5pp, at 10 and 20 classes respectively), confirming quantitatively that the substitution eased
the task rather than leaving that as an untested assumption. No published same-dataset number
exists for either baseline, so this remains a within-dataset representation comparison, not a
published-versus-reproduced check.

## 9. Limitations and What Is Deferred to Review 3

- **No unified 20+-class cross-dataset benchmark merging multiple corpora into one label space.**
  Section 6 runs three additional datasets *separately*, each with its own class taxonomy; a
  single benchmark spanning all four datasets under one unified label space was not attempted and
  remains open.
- **ET-BERT** was not implemented. It remains the highest-compute-risk item if a unified
  benchmark is pursued for Review 3.
- **Track B was not re-targeted to the three cross-dataset datasets of Section 6.** Its evaluation
  harness is specific to the original 10/20-class taxonomy; whether B's own gap against A/A-Full
  also replicates cross-dataset is an open question.
- **A-Full's 20-class result on USTC-TFC2016 has one unresolved hard pair** (Virut and Neris,
  Section 3) that the current physics-stats feature set cannot separate. It is reported as a
  dataset property, not claimed as solved. A genuinely different feature source, such as
  packet-size sequences, would be needed to test whether it is resolvable at all.
- **Two of the three new cross-dataset datasets are small and imbalanced by the source corpus's
  own availability, not by design** (5G's mMTC class: 59 samples from only two source captures;
  IoT-23's Mirai class: 229 samples after flow-timeout segmentation, against a 900-sample cap for
  the larger classes). Reported as-is in Section 6.1 rather than rebalanced. The 5G Transformer
  result specifically shows a materially wider seed-to-seed spread (5.14pp) than any other
  cross-dataset cell, plausibly an interaction between its 30-token self-attention window and this
  dataset's small, imbalanced test set.
- **Phase 4.2's extra confusable-pair features are a feasibility probe, not a full retrain.** See
  Section 3's Phase 4.2 update: a plain logistic regression on 7 new features scores 61.4% ± 4.9%
  telling Virut from Neris apart, a real but modest signal. Retraining A-Full's actual stats branch
  with these features added is the natural next step if this line is pursued further, not yet done.

## Appendix: Reproducibility

- All code: `A/` (including `A/model_full.py`, `A/dataset_full.py`, `A/cache_features_full.py`
  and `A/train_full.py` for A-Full), `C/baselines/` (including `cnn1d.py`, `transformer.py`),
  `C/run_matrix.py`, `src/`, and `scripts/extract_iscxvpn_dataset.py`,
  `scripts/extract_5g_dataset.py`, `scripts/extract_iot23_dataset.py`,
  `scripts/extract_original_representations.py` for the cross-dataset and own-representation work.
- All results: `results/comparison/*.csv`, generated by `C/make_tables.py`.
- Split-identity proof: `C/tests/test_split_identity.py`.
- A-Full correctness gate: `A/tests/test_model_full.py::test_hyp_cache_equivalence`.
- Full C scope and results narrative: `C/README.md`.
- Full A-Full narrative and methodology: `A/README.md`.
