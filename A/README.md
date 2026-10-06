# A: an independent TrafficCLIP implementation

Independent implementation of Chen et al., *"TrafficCLIP: A lightweight
cross-modal framework for network traffic classification"*, Computer Networks
272 (2025) 111662. Built from Algorithm 1 and Figs. 2-4 of that paper, and
sharing no code with `src/models/` (B). Keeping the two trees separate is the
point: if A imported B, B's conv1 initialisation would propagate into the
baseline and there would be nothing left to compare.

```
grep -rn "from src" A/   # must return nothing
```

## Architecture

| Module | File | Paper reference |
|---|---|---|
| Detail-aware encoder | `model.py:DetailAwareEncoder` | Fig. 3 |
| Semantics-aware encoder | `model.py:SemanticsAwareEncoder` | Sec. 4.1, Fig. 2 |
| Traffic adapter | `model.py:TrafficAdapter` | Eq. 3, Fig. 4 |
| Fusion + adapter residual | `model.py:TrafficCLIP.get_vision_features` | Eq. 4-5 |
| Text encoder (frozen BERT, static hard prompt) | `model.py:TrafficCLIP.get_text_features` | Sec. 4.2, Eq. 6 |
| Cosine similarity x learned temperature | `model.py:TrafficCLIP.forward` | Eq. 7 |
| Loss (CE + contrastive) | `train.py` | Eq. 8-9 |

## The main divergence from B: grayscale conv1

Pretrained ResNet-50's `conv1` is a `(64, 3, 7, 7)` kernel trained for
3-channel RGB input, while USTC-TFC2016 samples are 1-channel 28x28 byte
images. B (`src/models/traffic_clip.py:106`) discards this kernel and replaces
it with a fresh, randomly initialised `Conv2d(1, 64, ...)` that then stays
frozen, so the pretrained ResNet-50 sees nothing but noise through its first
layer. A adapts the kernel instead (`model.py:adapt_conv1_to_grayscale`), by
summing the pretrained weights across the input-channel dimension into a
`(64, 1, 7, 7)` kernel.

This is not an approximation. It is the exact grayscale case of the standard
replicate-to-3-channels adaptation: for a grayscale value `G` replicated to
`R=G=B=G` and convolved with the original 3-channel kernel, output channel `o`
at each position is `sum_c(W[o,c] * G) = (sum_c W[o,c]) * G`. Summing the kernel
first gives the identical result in one pass instead of three.
`A/tests/test_model.py::test_conv1_adaptation_matches_replicated_grayscale_forward`
checks this numerically at `atol=1e-4` against an unmodified pretrained
ResNet-50.

`--broken-conv1` reinstalls B's version verbatim, a fresh random frozen
`Conv2d(1, 64, 7, 7)`, inside A's otherwise-correct codebase, so its
contribution can be measured on its own. Everything else about the model is
identical between the default and the ablation.

## Other choices, and why

- **Static prompt only, no per-sample dynamic prompt.** The base paper's text
  side is one hard prompt per class name (Sec. 4.2). It has no physics-informed
  per-sample prompt over IAT, jitter and entropy anywhere; that is a B/C
  addition (`src/dataset.py`'s `use_dynamic_prompts`). As a result A's training
  objective and its evaluation, hypothesis testing over K candidate prompts, are
  the *same computation*: comparing each image's fused vision embedding against
  the same K static text embeddings and taking the argmax. There is no separate
  evaluation path that could see the label.
- **Split fixed once per archive, not per seed** (`dataset.py:SPLIT_SEED`). The
  paper's "three independent runs, averaged" varies model init and batch order,
  not the data partition. Regenerating the split per seed would leak the test
  set across runs.
- **Frozen submodules pinned to eval mode inside `TrafficCLIP.train()`**
  (`model.py`), rather than left to each call site. `model.train()` on the whole
  model otherwise leaves BERT's dropout active during a live forward pass, which
  is a hard crash on MPS (`scaled_dot_product_attention` has no dropout kernel
  there) and silent non-determinism elsewhere, and it lets ResNet-50's BatchNorm
  running stats drift despite `requires_grad=False`, which blocks gradients
  rather than `.training`.
- **Feature cache, built separately from B's** (`cache_features.py`,
  `A/cache/<archive>/<correct_conv1|broken_conv1>/`). `semantics_encoder` and
  `text_encoder` are frozen and receive no augmentation, so their outputs are
  constant per sample and per class across every epoch and seed. Caching them is
  exact rather than approximate (see `docs/feature_cache.md` for the same
  argument applied to B). A's cache is much smaller than B's: because text is K
  static prompts rather than a per-sample dynamic one, there is no
  `bert_pooler_train` or `bert_pooler_hyp`, the entire text side is one
  `(K, 768)` tensor, and building it takes seconds rather than 10 to 35 minutes.
  `A/tests/test_model.py::test_cache_equivalence` is the gate: if cached and
  live vision features disagree, every downstream number is wrong.
- **Contrastive loss** (`train.py:supervised_contrastive_loss`) follows the
  standard SupCon convention (Khosla et al., 2020) of excluding the sample
  itself from both the positive set and the denominator sum. The paper's Eq. 9
  states the index bounds (`a=1..N`) without resolving the self-term. A batch
  where a sample's class happens to be a singleton, which happens occasionally
  at 20 classes and batch 64, contributes 0 for that sample.
- **Loss weighting: unweighted sum**, `L = L_CE + L_CL`, exactly as Algorithm 1
  states it (`Total_loss = L_ce + L_cl`, no lambda term). The paper never gives
  a `lambda_CL` value in the body text either.
- **SGD momentum 0.9.** The paper specifies SGD, lr 0.002, cosine annealing and
  a one-epoch 1e-5 warm-up (Sec. 5.1.4), but not momentum. 0.9 is the
  conventional default; it is not a paper-stated value.
- **Detail-aware encoder: 2 residual blocks**, `mid_channels=16`. The paper says
  "multiple residual blocks" without a count; this is not a paper-stated value.

## Usage

```bash
export PYTHONPATH=.

# Build the feature cache once per archive (and per conv1 variant).
.venv/bin/python A/cache_features.py --npz data/processed/ustc_10class.npz
.venv/bin/python A/cache_features.py --npz data/processed/ustc_20class.npz
.venv/bin/python A/cache_features.py --npz data/processed/ustc_20class.npz --broken-conv1

# Train: 3 seeds, 200 epochs, writes results/comparison/A_<tag>.csv and
# A/checkpoints/A_<tag>_seed<seed>.pt
.venv/bin/python A/train.py --npz data/processed/ustc_10class.npz --classes 10
.venv/bin/python A/train.py --npz data/processed/ustc_20class.npz --classes 20
.venv/bin/python A/train.py --npz data/processed/ustc_20class.npz --classes 20 --broken-conv1

# Or run the whole matrix sequentially (MPS is one GPU; don't parallelize):
bash scripts/run_a_matrix.sh

# Check progress on a running matrix (reads the JSON A/train.py writes every
# epoch, so there's no need to watch the logs):
bash scripts/progress.sh

# Re-evaluate a saved checkpoint without retraining:
.venv/bin/python A/evaluate.py --checkpoint A/checkpoints/A_20class_seed62.pt \
    --npz data/processed/ustc_20class.npz

# Tests (shape, gradient flow, conv1 adaptation, cache equivalence, loss/scheduler):
.venv/bin/python -m pytest A/tests/ -v
```

## Results (M4, MPS, 3 seeds x 200 epochs each)

| Run | Accuracy | Macro F1 | CSV |
|---|---|---|---|
| A, 10-class | 99.75% ± 0.09pp | 99.75% ± 0.09pp | `results/comparison/A_10class.csv` |
| A, 20-class | 96.64% ± 0.23pp | 96.64% ± 0.23pp | `results/comparison/A_20class.csv` |
| A, 20-class, `--broken-conv1` ablation | 96.68% ± 0.14pp | 96.68% ± 0.14pp | `results/comparison/A_20class_brokenconv1.csv` |

**The broken-conv1 ablation does not reproduce B's ~62%.** It scores 96.68%,
statistically indistinguishable from the correct-conv1 run and if anything a
hair higher. Isolated inside an otherwise correct codebase, the conv1
initialisation costs approximately nothing. The likely explanation is that the
fully trainable detail-aware encoder contributes half of `V_fused` regardless of
what the frozen semantics branch produces, and the trainable `semantics_proj`
and `visual_adapter` sit between the frozen backbone and the loss, so both
pathways have enough capacity to compensate for a randomized first conv layer.
That overturns the original leading hypothesis that conv1 alone drives B's gap
against Chen et al.'s claimed 99.05%. B's ~62% needs its own explanation from
B's own measurements, not an inherited one from this ablation.

## Interpreting the results

`results/comparison/A_20class.csv` is the row to compare against the paper's
claimed 99.05% (Table 1). Where a gap persists after the conv1 fix, the
remaining divergences worth documenting are, in rough order of expected impact:
the paper's implementation is CPU/CUDA on two RTX 4090s where this runs on an
M4's MPS backend (no known numerical divergence, but untested at this scale by
the paper's authors); detail-encoder block count and width are unspecified in
the paper and guessed here; the exact payload-to-image byte-masking granularity
is ambiguous (`docs/dataset_schema.md` covers flow-level against per-packet
masking); and three runs are averaged against an unspecified initialization
scheme for those runs.

`results/comparison/A_20class_brokenconv1.csv` isolates the conv1 initialisation
on its own. The gap between it and `A_20class.csv` answers "how much did the
random frozen conv1 cost B", independent of every other difference between the
two codebases.

## A-Full: does the tri-modal design close B's gap?

`model_full.py`, `dataset_full.py`, `cache_features_full.py` and `train_full.py`
extend the above with the two OptimizedTrafficCLIP-only additions: a
physics-stats branch over IAT, jitter and entropy, and per-sample dynamic
prompts, fused through a non-linear MLP head instead of cosine similarity. That
is B's architecture, implemented independently here to avoid B's known code
defects by construction (see `model_full.py`'s module docstring). The question
is whether B's honest-eval gap, 88.13% (5-seed mean) against its manuscript's
claims, comes from remaining code defects or from something inherent to the
tri-modal design.

**One real engineering problem surfaced while building it.** B's training loop
feeds the sample's own true-label prompt as input, so the model never has to
disambiguate between candidates during training. A-Full instead trains with the
same honest K-hypothesis computation used at eval time
(`TrafficCLIPFull.hypothesis_logits`), matching base A's train/eval symmetry. A
naive live-BERT version of that measured ~7.5s/step (B x K = 1280 BERT forwards
per step), which projects to ~246 hours for the full 200-epoch, 3-seed matrix.
Since every sample's real stats are fixed for the whole run, its K
dynamic-prompt embeddings are too, so `cache_features_full.py` precomputes them
once into an `(N, K, 768)` cache (~1.1 GiB at N=18000, K=20), cutting the same
matrix to about an hour. `A/tests/test_model_full.py`'s
`test_hyp_cache_equivalence` is the correctness gate for it, mirroring
`test_cache_equivalence` for the base cache.

```bash
PYTHONPATH=. .venv/bin/python A/cache_features_full.py --npz data/processed/ustc_10class.npz
PYTHONPATH=. .venv/bin/python A/cache_features_full.py --npz data/processed/ustc_20class.npz
PYTHONPATH=. .venv/bin/python A/train_full.py --npz data/processed/ustc_10class.npz --classes 10
PYTHONPATH=. .venv/bin/python A/train_full.py --npz data/processed/ustc_20class.npz --classes 20
PYTHONPATH=. .venv/bin/python A/train_full.py --npz data/processed/ustc_20class.npz --classes 20 --broken-conv1
```

### Results (M4, MPS, 3 seeds x 200 epochs)

| Run | Accuracy | Macro F1 | CSV |
|---|---|---|---|
| A-Full, 10-class (stats branch + dynamic prompts + MLP fusion) | 99.68% ± 0.07pp | 99.68% ± 0.07pp | `results/comparison/A_Full_10class.csv` |
| A-Full, 20-class | 96.63% ± 0.09pp | 96.64% ± 0.08pp | `results/comparison/A_Full_20class.csv` |
| A-Full, 20-class, `--broken-conv1` ablation | 96.80% ± 0.20pp | 96.79% ± 0.21pp | `results/comparison/A_Full_20class_brokenconv1.csv` |
| A, 10-class (for comparison, no stats/dynamic prompts) | 99.75% ± 0.09pp | 99.75% ± 0.09pp | `results/comparison/A_10class.csv` |
| A, 20-class (for comparison, no stats/dynamic prompts) | 96.64% ± 0.23pp | 96.64% ± 0.23pp | `results/comparison/A_20class.csv` |
| A, 20-class, `--broken-conv1` ablation | 96.68% ± 0.14pp | 96.68% ± 0.14pp | `results/comparison/A_20class_brokenconv1.csv` |
| B, honest "Full" (same tri-modal design, B's own code, 5-seed mean) | 88.13% ± 2.12pp | 88.29% ± 2.15pp | see `paper/review2_findings.md` |

**The answer is code, not architecture.** A-Full essentially matches base A,
which has no stats branch and no dynamic prompts, so on this dataset the
tri-modal additions cost nothing and do not obviously help either. But both
independent variants land about 8.5pp above B's own honest score for the *same*
tri-modal design. Once the implementation defects are out of the way, the
physics-stats-plus-dynamic-prompt architecture is not what holds B back; B's
implementation of it is. This does not settle whether the added complexity is
*worth it* in general, since USTC-TFC2016 may simply be too easy a dataset for a
tri-modal model's real advantages, on obfuscated traffic, unseen classes or
zero-shot, to show up at all when everything solves it near ceiling. It only
shows that B's 62-85% range is not an architectural ceiling.

**The `--broken-conv1` ablation costs nothing here either**: 96.80% against
96.63% with the correct conv1, statistically indistinguishable, and the same
pattern as base A's own ablation (96.68% against 96.64%). All four variants, A
and A-Full crossed with correct and broken conv1, cluster at 96.6-96.8%, while B
sits about 8.5pp below every one of them. B's gap is not any single isolated
architectural defect, including the conv1 initialisation originally hypothesized
as the leading cause.

**The parity result is not specific to USTC-TFC2016.** Both A and A-Full were
re-run, unmodified beyond `--classes`, on three further independently sourced
datasets (ISCXVPN2016, a 5G traffic dataset, IoT-23; see
`paper/review2_findings.md` Section 6). A and A-Full remain statistically
indistinguishable on all three, so the finding that the tri-modal additions
neither help nor hurt is not an artifact of one dataset's own separability.

### Where the remaining 20-class error comes from

A per-class breakdown of A-Full's 20-class test predictions (3 seeds, correct
conv1) shows the ~3.4% error is not spread across the dataset. 18 of 20 classes
score 95-100%, most of them exactly 100%, and one pair, **Virut and Neris**,
both botnet malware families, accounts for 75.8% of all misclassifications
summed across the 3 seeds (207 of 273 errors). Per seed this ranges from 70.5%
(seed 64) to 82.4% (seed 62) to 74.5% (seed 63) -- a real spread rather than
flat consistency; see `scripts/analyze_confusable_pair.py` and
`paper/review2_findings.md`'s "Anatomy of a confusable pair" for the full
breakdown. Every other class is effectively solved (Nsis-ay is the only other
class below 97.8%, at 95.1%).

The stats branch carries no signal for this pair: their (IAT, jitter, entropy)
distributions are nearly identical at the median (IAT 1793 against 1783ms,
jitter 2399 against 2396ms, entropy 4.94 against 4.93), and the large mean and
std divergence is outlier-driven rather than a real separation. Whatever the
model uses to tell them apart comes from the vision branch, not the tri-modal
addition.

**This is a dataset property, not a code defect.** Virut/Neris confusion in
USTC-TFC2016 is a known-hard pair elsewhere in the traffic-classification
literature, and the feature set here (mean IAT, jitter, byte entropy) does not
distinguish them. There is no bug to fix and no branch to add without a
genuinely different feature source, such as packet-size sequences or protocol
header fields, neither of which this project extracts. So A-Full's 96.6% on
20-class is a near-solved problem, 18 of 20 classes at ceiling, with one
well-understood hard pair, rather than a general architectural weakness.
