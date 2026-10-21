# Dataset schema

Produced by `src/make_traffic_dataset.py`. One `.npz` per class set.

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `x` | `(N, 1, 28, 28)` | `float32` | Flow image. First 784 bytes of the flow, header-masked, divided by 255. |
| `y` | `(N,)` | `int64` | Class index into `labels`. |
| `m` | `(N, 3)` | `float64` | `[mean_IAT_ms, jitter_ms, byte_entropy_bits]`. |
| `labels` | `(K,)` | `<U` | **K class names, indexed by `y`.** Not per-sample. |

## `labels` is (K,), not (N,)

The original project's run book (section 3.2, not included in this package)
documents `labels` as an `(N,)` array carrying one name per sample. It is not. The writer emits
`list(label_map.keys())`, one name per class, positionally aligned with `y`:

```python
class_name = data["labels"][y[i]]     # correct
class_name = data["labels"][i]        # what the guide implies; wrong
```

`src/dataset.py` (`self.class_names[label]`) and `src/benchmark.py` both depend
on the real behaviour, so the schema stays as it is and the guide is the thing
that is wrong. Treat this file as the reference for the schema.

## Archives

| File | Classes | Samples/class | Total |
|---|---|---|---|
| `data/processed/ustc_10class.npz` | the manuscript's 10-class subset | 900 | 9,000 |
| `data/processed/ustc_20class.npz` | all 20 USTC-TFC2016 classes | 900 | 18,000 |
| `data/processed/traffic_dataset_metadata_mini.npz` | 10-class subset | 10 | 100 |

Labels are assigned over the **sorted** class-folder names of each archive
independently, so index 3 in the 10-class archive and index 3 in the 20-class
archive are different classes. Always read the class name through `labels`.

## Class sets

The 10-class subset is the manuscript's (section V.A): five benign (Skype,
MySQL, BitTorrent, Facetime, Weibo) and five malware (Zeus, Tinba, Cridex,
Geodo, Miuref). The 20-class set adds FTP, Gmail, Outlook, SMB,
WorldOfWarcraft, Htbot, Neris, Nsis-ay, Shifu, Virut.

**Gmail** is in the 20-class set but not the 10-class subset, despite appearing
in `src/run_explain.sh`'s `--conflicts` default and in the commented-out
`prompts:` block of `config/config.yaml`. Those belong to a superseded six-class
experiment, {BitTorrent, Cridex, Gmail, Skype, Tinba, Zeus} at 1500
samples/class, from the original project's v0.0.1 release (its results are not
included in this package; the class names on its confusion-matrix axes identify
the class set). The manuscript replaced that experiment with the 10-class subset
above.

## Header masking

`--mask-mode` controls masking granularity:

- `flow` (default) zeroes bytes 0-53 of the **concatenated flow**, so packets
  2..n keep their headers. This is the original behaviour, and it is the
  default so that B stays comparable with the manuscript.
- `packet` zeroes each packet's own 54-byte header before concatenation. This
  is what the base paper describes, and what A uses.
- `none` does no masking.

Statistics in `m` are computed over **unmasked** bytes in all three modes,
matching the original ordering of stats before masking.

## Flow extraction

Flows are keyed on the bidirectional 5-tuple with endpoints sorted as
`(ip, port)` **pairs**. The original sorted addresses and ports independently,
which merged distinct conversations onto one key. See `tests/test_flow_key.py`.

Within a class, flows are taken in order of first packet appearance, capture
files in sorted filename order, until `samples_per_class` is reached.

---

## Flows are never timed out

Found while verifying the built archives, and it bears directly on the
manuscript's headline contribution.

`flow_key` groups every packet sharing a bidirectional 5-tuple into one flow,
with **no idle timeout**. Real flow extraction splits a conversation when the
gap between packets exceeds an inactivity threshold, conventionally 60 seconds.
Without one, packets from the same endpoints hours apart are concatenated into a
single "flow", and `mean_IAT` stops describing traffic behaviour and starts
describing how long the capture ran.

### Measured on the built 10-class archive

| Class | median `mean_IAT` | max | samples > 1 h |
|---|---|---|---|
| BitTorrent | 0.06 ms | 0.2 ms | 0 |
| **Cridex** | **12,190,082 ms (3.4 h)** | 52,085,703 ms | **895 / 900** |
| **Facetime** | **0.00 ms** | 0.00 ms | 0 |
| Geodo | 4,499.69 ms | 95,984,224 ms | 2 |
| Miuref | 4,501.13 ms | 108,001,346 ms | 2 |
| MySQL | 0.03 ms | 0.1 ms | 0 |
| Skype | 62.76 ms | 316.2 ms | 0 |
| Tinba | 22.24 ms | 1,357,418 ms | 0 |
| Weibo | 0.03 ms | 2.7 ms | 0 |
| Zeus | 195.60 ms | 6,387,397 ms | 45 |

944 of 9,000 samples (10.5%) carry a `mean_IAT` above one hour.

**Cridex.pcap** spans 194 hours of wall clock. 16,378 of its 16,385 flows span
more than an hour, and the median flow is 12 packets. A 60-second idle timeout
splits those 16,385 flows into 60,058 segments and drops the median `mean_IAT`
from 3.4 hours to **263 ms**, with no segment above an hour.

**Facetime.pcap** is the opposite degeneracy: 6,000 packets produce 6,000 flows,
**all of them single-packet**, so every Facetime sample has `mean_IAT = 0.0`
and `jitter = 0.0` exactly.

### Why it matters

The physics modality, `[mean_IAT, jitter, entropy]`, is the paper's central
addition to TrafficCLIP, and it is a **shortcut feature** for at least two of
the ten classes. Cridex is identifiable from the sheer magnitude of `mean_IAT`,
Facetime from its being exactly zero. Neither reflects an application's
behaviour; both reflect how the capture was recorded and how the flows were not
segmented.

It reaches the text branch too. The dynamic prompt for the worst sample reads:

> "A network traffic gray photo of class Miuref with 108001346.76ms mean IAT,
> 69483726.98ms jitter, and 5.06 byte entropy."

A 12-digit millisecond figure is tokenized by BERT into arbitrary word-pieces,
and `StatsProjectionHead` receives these raw and unnormalized, across eight
orders of magnitude.

### What was done

`--flow-timeout` defaults to **0**, i.e. disabled, which reproduces the original
behaviour and keeps B faithful to the manuscript. A 60-second variant is built
alongside it:

```bash
PYTHONPATH=. .venv/bin/python src/make_traffic_dataset.py --variant all
PYTHONPATH=. .venv/bin/python src/make_traffic_dataset.py --variant all --flow-timeout 60 --suffix _t60
```

giving `ustc_{10,20}class.npz` and `ustc_{10,20}class_t60.npz`. The pair
isolates this effect the same way `--broken-conv1` isolates the conv1
initialisation.

## All-zero images

Three of the 18,000 twenty-class samples, all Neris, were single header-only
packets, a bare SYN/ACK with no payload, so the 54-byte mask consumed the entire
image and left pure black. `--keep-empty` retains them; the default drops them
and takes the next available flow instead. None fall in the 10-class subset, so
B is unaffected either way.
