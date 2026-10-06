# Environment

**Target machine:** MacBook Air M4, 16 GB unified memory, macOS (arm64).

## Setup

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Experiment tracking and ONNX export are a separate install:

```bash
uv pip install --python .venv/bin/python -r requirements-tracking.txt
```

Run everything through `.venv/bin/python` from the project root, so `src` is
importable:

```bash
.venv/bin/python -m pytest
.venv/bin/python src/make_traffic_dataset.py --variant all
```

## Why not miniforge

`uv` was already installed, every dependency in `environment.yml` is on PyPI,
and pip's `osx-arm64` torch wheels have MPS compiled in, so conda buys nothing
here and costs about 500 MB. The conda spec is still maintained, and now
actually solvable, for anyone following the original run book.

## Device selection

`src/utils/utils.get_device()` prefers CUDA, then Apple MPS, then CPU. Every
entry point previously hardcoded `torch.device("cuda" if torch.cuda.is_available()
else "cpu")`, which resolves to CPU on Apple silicon, so the GPU was never used
in any run.

Pass `get_device(prefer="cpu")` where CPU is required. The ONNX export and
quantization paths in `src/optimization/profiling.py` must stay on CPU: the MPS
backend does not implement the quantized operators, and
`torch.quantization.QuantStub` in `opt_traffic_clip.py` is inert there.

`set_seed()` guards its CUDA calls and seeds MPS when present.

## Hugging Face downloads hang on the xet backend

The first `AutoModel.from_pretrained("google-bert/bert-base-uncased")` or
`AutoTokenizer` call against a cold cache can hang indefinitely partway through
downloading `model.safetensors`, stuck requesting an `xet-read-token` and never
starting the transfer. Set `HF_HUB_DISABLE_XET=1` to force the plain HTTP
download path:

```bash
HF_HUB_DISABLE_XET=1 PYTHONPATH=. .venv/bin/python src/cache_features.py --npz data/processed/ustc_10class.npz
```

Once `~/.cache/huggingface/hub/models--google-bert--bert-base-uncased/` is
populated, later runs load from disk and never touch the network, so the flag
is only needed for the first download.

## Verified versions

| Package | Version |
|---|---|
| Python | 3.12.14 |
| torch | 2.14.0 |
| torchvision | 0.29.0 |
| transformers | 5.16.1 |
| numpy | 2.5.2 |

`torch.backends.mps.is_available()` returns `True` and `get_device()` returns
`mps`.
