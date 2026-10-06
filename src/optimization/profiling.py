import argparse
import logging
import sys
import time
from pathlib import Path

import cloudpickle
import mlflow
import numpy as np
import onnxruntime as ort
import torch
from sklearn.metrics import f1_score

from transformers import AutoTokenizer

from src.models.opt_traffic_clip import OptimizedTrafficCLIP
from src.optimization.calibration import create_profiling_dataloader
from src.optimization.quant import export_to_onnx
from src.utils.utils import load_config

# def profile_model(model, num_samples=100, warm_up=10):
#     """
#     Measures Inference Latency and Throughput on CPU.
#     """
#     # Ensure the model is in evaluation mode and on CPU
#     model.eval()
#     model.to("cpu")

#     # Create dummy inputs for Tri-modal shapes
#     # (Batch, Channel, H, W) -> (1, 1, 28, 28) for images
#     dummy_img = torch.randn(1, 1, 28, 28)
#     # (Batch, Stats) -> (1, 3) for [Mean IAT, Jitter, Entropy]
#     dummy_stats = torch.randn(1, 3)
#     # For text, we simulate a single encoded prompt vector
#     dummy_text = torch.randn(1, 1024)

#     logger.info("Starting Profiling")

#     # 1. Warm-up Phase: "Wakes up" the CPU and fills the cache
#     with torch.no_grad():
#         for _ in range(warm_up):
#             _ = model(dummy_img, dummy_stats, dummy_text)

#     # 2. Measurement Phase
#     latencies = []
#     with torch.no_grad():
#         for i in range(num_samples):
#             start_time = time.perf_counter()
#             _ = model(dummy_img, dummy_stats, dummy_text)
#             end_time = time.perf_counter()

#             # Convert to milliseconds
#             latencies.append((end_time - start_time) * 1000)

#     # 3. Calculate Metrics
#     avg_latency = np.mean(latencies)
#     p99_latency = np.percentile(latencies, 99)  # Tail latency
#     throughput = 1000 / avg_latency  # Flows per second

#     logger.info(f"Average Latency: {avg_latency:.2f} ms")
#     logger.info(f"P99 (Tail) Latency: {p99_latency:.2f} ms")
#     logger.info(f"Throughput: {throughput:.2f} flows/sec")
#     logger.info(f"---------------------------------")

#     return avg_latency, throughput


# if __name__ == "__main__":
#     from src.models.opt_traffic_clip import OptimizedTrafficCLIP

#     config = load_config()
#     traffic_cfg = config["dataset"]["traffic"]["classes"]

#     num_classes = sum(len(class_list) for class_list in traffic_cfg.values())
#     model = OptimizedTrafficCLIP(num_classes=num_classes, use_stats=False)
#     model.load_state_dict(torch.load("best_trafficclip_model.pt", map_location="cpu"))

#     avg_latency, throughput = profile_model(model)


# def evaluate_quantized_system(
#     original_model, quantized_model, test_loader, original_path, quantized_path
# ):
#     """
#     Enhanced evaluation that merges accuracy testing with professional
#     profiling (warm-up, P99, and throughput).
#     """
#     logger.basicConfig(level=logger.INFO, format="%(message)s")

#     def run_comprehensive_bench(model, loader, model_name):
#         model.eval()
#         model.to("cpu")

#         all_preds = []
#         all_labels = []
#         latencies = []

#         # 1. Warm-up Phase (Critical for realistic CPU metrics)
#         # We process a small batch to wake up the CPU and fill the L3 cache
#         warm_up_batches = 5
#         with torch.no_grad():
#             for i, (img, stats, ids, mask, _) in enumerate(loader):
#                 if i >= warm_up_batches:
#                     break
#                 _ = model(img, ids, mask, stats)

#         # 2. Combined Accuracy & Latency Measurement
#         logger.info(f"Benchmarking {model_name}...")
#         with torch.no_grad():
#             for img, stats, ids, mask, labels in loader:
#                 # We measure latency per batch to get a distribution for P99
#                 start = time.perf_counter()
#                 logits, _ = model(img, ids, mask, stats)
#                 end = time.perf_counter()

#                 # Per-sample latency in this batch
#                 batch_latency = ((end - start) * 1000) / img.size(0)
#                 latencies.append(batch_latency)

#                 preds = torch.argmax(logits, dim=1)
#                 all_preds.extend(preds.numpy())
#                 all_labels.extend(labels.numpy())

#         # 3. Calculate Metrics
#         f1 = f1_score(all_labels, all_preds, average="macro")
#         avg_lat = np.mean(latencies)
#         p99_lat = np.percentile(latencies, 99)
#         throughput = 1000 / avg_lat  # Flows per second per core

#         return f1, avg_lat, p99_lat, throughput

#     # Run benchmarks
#     orig_f1, orig_lat, orig_p99, orig_tp = run_comprehensive_bench(
#         original_model, test_loader, "Float32 Model"
#     )
#     quant_f1, quant_lat, quant_p99, quant_tp = run_comprehensive_bench(
#         quantized_model, test_loader, "Int8 Model"
#     )

#     # 4. Model Size Calculation
#     orig_size = os.path.getsize(original_path) / (1024 * 1024)
#     quant_size = os.path.getsize(quantized_path) / (1024 * 1024)

#     # --- FINAL APPLIED SCIENCE REPORT ---
#     print("\n" + "=" * 50)
#     print("      TRAFFIC-CLIP PRODUCTION REPORT")
#     print("=" * 50)
#     print(f"{'Metric':<20} | {'Original (FP32)':<12} | {'Quantized (INT8)':<12}")
#     print(f"{'-'*20}-|-{'-'*12}-|-{'-'*12}")
#     print(f"{'Model Size (MB)':<20} | {orig_size:<12.2f} | {quant_size:<12.2f}")
#     print(f"{'Macro F1 Score':<20} | {orig_f1:<12.4f} | {quant_f1:<12.4f}")
#     print(f"{'Avg Latency (ms)':<20} | {orig_lat:<12.2f} | {quant_lat:<12.2f}")
#     print(f"{'P99 Latency (ms)':<20} | {orig_p99:<12.2f} | {quant_p99:<12.2f}")
#     print(f"{'Throughput (fps)':<20} | {orig_tp:<12.1f} | {quant_tp:<12.1f}")
#     print(f"{'-'*50}")
#     print(
#         f"RESULTS: {orig_lat/quant_lat:.1f}x Speedup | {orig_size/quant_size:.1f}x Compression"
#     )
#     print("=" * 50)


#     return quant_f1, quant_lat
def get_model_size_mb(model_object):
    """
    Calculates the size of a model object in MB using cloudpickle.
    """
    # Serialize the model object to bytes
    model_bytes = cloudpickle.dumps(model_object)

    # Calculate size in Megabytes
    size_mb = len(model_bytes) / (1024 * 1024)
    return size_mb


def _build_hypothesis_batch(batch, class_names, tokenizer, max_length, use_stats):
    """
    Defect 2 fix: expands a batch of B samples into B*K (image, candidate-class
    text) pairs for label-free hypothesis-testing prediction -- the same
    protocol `validate()` uses (src/train.py). The previous version predicted
    with argmax() of a single forward pass whose text input was the sample's
    own dynamic prompt, built from its *true* label
    (`TrafficDataset.__getitem__`, src/dataset.py) -- not a prediction, since
    the ground truth was already in the input the "classifier" read.
    """
    images = batch["image"]
    stats = batch["stats"]
    b = images.size(0)
    k = len(class_names)
    stats_np = stats.numpy()

    prompts = [
        f"A network traffic gray photo of class {name} with "
        f"{stats_np[i, 0]:.2f}ms mean IAT, {stats_np[i, 1]:.2f}ms jitter, "
        f"and {stats_np[i, 2]:.2f} byte entropy."
        for i in range(b)
        for name in class_names
    ]
    encoded = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )

    images_rep = (
        images.unsqueeze(1)
        .expand(b, k, *images.shape[1:])
        .reshape(b * k, *images.shape[1:])
    )
    hyp_batch = {
        "image": images_rep,
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }
    if use_stats:
        hyp_batch["stats"] = (
            stats.unsqueeze(1).expand(b, k, stats.shape[1]).reshape(b * k, stats.shape[1])
        )
    return hyp_batch, b, k


def evaluate_quantized_system(
    original_model, optimized_model, test_loader, class_names, tokenizer, max_length, use_stats
):

    def run_comprehensive_bench(model, loader, model_name):
        is_ort = isinstance(model, ort.InferenceSession)

        if not is_ort:
            model.eval()
            model.to("cpu")
        else:
            onnx_inputs = [i.name for i in model.get_inputs()]
            logger.info(f"{model_name} ONNX inputs: {onnx_inputs}")

        all_preds, all_labels, latencies = [], [], []
        warm_up_batches = 5

        def predict(batch):
            """Hypothesis-test this batch; returns (preds[B], batch_size B)."""
            hyp_batch, b, k = _build_hypothesis_batch(
                batch, class_names, tokenizer, max_length, use_stats
            )
            if is_ort:
                feeds = {name: hyp_batch[name].numpy() for name in onnx_inputs}
                outputs = model.run(None, feeds)
                logits = torch.from_numpy(outputs[0])
            else:
                if use_stats:
                    logits, _ = model(
                        hyp_batch["image"],
                        hyp_batch["input_ids"],
                        hyp_batch["attention_mask"],
                        hyp_batch["stats"],
                    )
                else:
                    logits, _ = model(
                        hyp_batch["image"],
                        hyp_batch["input_ids"],
                        hyp_batch["attention_mask"],
                    )
            # [B*K, K_classes] -> [B, K, K_classes]; the predicted class per
            # sample is the hypothesis whose own class-index logit is
            # highest (the batched equivalent of the per-sample torch.diag()
            # in validate()).
            logits = logits.view(b, k, -1)
            confidences = torch.diagonal(logits, dim1=1, dim2=2)  # [B, K]
            preds = torch.argmax(confidences, dim=1)
            return preds, b

        logger.info(f"Warming up {model_name}...")

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if i >= warm_up_batches:
                    break
                predict(batch)

        logger.info(f"Benchmarking {model_name}...")

        with torch.no_grad():
            for batch in loader:
                labels = batch["label"]

                start = time.perf_counter()
                preds, b = predict(batch)
                end = time.perf_counter()

                # Per-sample latency for the corrected protocol legitimately
                # includes the K-way hypothesis test -- K times the compute
                # of the leaky single-forward-pass baseline. That cost
                # difference is a real finding, not an artifact to hide.
                latencies.append(((end - start) * 1000) / b)

                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds, average="macro")
        avg_lat = np.mean(latencies)
        p99_lat = np.percentile(latencies, 99)
        tp = 1000 / avg_lat

        return f1, avg_lat, p99_lat, tp

    # --- EXECUTION ---
    orig_f1, orig_lat, orig_p99, orig_tp = run_comprehensive_bench(
        original_model, test_loader, "PyTorch_FP32"
    )

    opt_f1, opt_lat, opt_p99, opt_tp = run_comprehensive_bench(
        optimized_model, test_loader, "ONNX_Optimized"
    )

    # --- LOGGING ---
    orig_size = get_model_size_mb(original_model)

    onnx_file = Path("traffic_clip_optimized.onnx")
    opt_size = (
        onnx_file.stat().st_size / (1024 * 1024) if onnx_file.exists() else orig_size
    )

    mlflow.log_metrics(
        {
            "orig_f1": orig_f1,
            "opt_f1": opt_f1,
            "orig_lat_ms": orig_lat,
            "opt_lat_ms": opt_lat,
            "orig_p99_ms": orig_p99,
            "opt_p99_ms": opt_p99,
            "speedup": orig_lat / opt_lat,
        }
    )

    logger.info("\n" + "=" * 60)
    logger.info(
        f"{'Metric':<20} | {'Baseline (PyTorch)':<18} | {'Optimized (ONNX)':<18}"
    )
    logger.info(f"{'-'*20}-|-{'-'*18}-|-{'-'*18}")
    logger.info(f"{'Macro F1 Score':<20} | {orig_f1:<18.4f} | {opt_f1:<18.4f}")
    logger.info(f"{'Avg Latency (ms)':<20} | {orig_lat:<18.2f} | {opt_lat:<18.2f}")
    logger.info(f"{'P99 Latency (ms)':<20} | {orig_p99:<18.2f} | {opt_p99:<18.2f}")
    logger.info(f"{'Throughput (FPS)':<20} | {orig_tp:<18.1f} | {opt_tp:<18.1f}")
    logger.info(f"{'Model Size (MB)':<20} | {orig_size:<18.2f} | {opt_size:<18.2f}")
    logger.info("=" * 60)

    return opt_f1, opt_lat


if __name__ == "__main__":
    # --- 1. Path & Environment Setup ---
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]

    # Ensure project root is in path for custom module discovery
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    # Register the model class for safe unpickling (PyTorch 2.6+)
    import torch.serialization

    from models.opt_traffic_clip import OptimizedTrafficCLIP

    torch.serialization.add_safe_globals([OptimizedTrafficCLIP])

    config = load_config()

    parser = argparse.ArgumentParser(
        description="Profile TrafficCLIP System Performance"
    )
    parser.add_argument("--lambda_cl", type=float, required=True)
    parser.add_argument("--use_stats_prompts", action="store_true")
    parser.add_argument("--model_version", type=str, default="optimized")
    parser.add_argument("--use_stats", action="store_true", default=False)
    parser.add_argument("--stats_input_dim", type=int, default=3)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a local .pt state_dict (e.g. from train.py's "
            "experiments/<version>/<model_type>/best_model.pt). If omitted, "
            "profiles a freshly-initialized model -- still exercises the "
            "full ONNX export + hypothesis-testing benchmark pipeline."
        ),
    )
    args = parser.parse_args()

    # Local-only rather than the DagsHub account in config.yaml's user.name;
    # see train_runner.py. Also, "models:/best_ht/latest" (the previous weight
    # source) is a model registry entry nothing in this codebase ever
    # registers. --checkpoint above replaces it with a plain local
    # state_dict load.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    logger = logging.getLogger("TrafficCLIP")  # Named logger
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        # Stream (Terminal)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)

        # File
        fh = logging.FileHandler("profile.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    mlflow.set_experiment("TrafficCLIP_Quantization")
    # --- 2. System Evaluation ---
    with mlflow.start_run(run_name="ONNX_System_Profiling"):
        try:
            # STEP A: Prepare Test Data & Dummy Batch first -- num_classes
            # for the model below must come from the npz actually loaded,
            # not from config["dataset"]["traffic"]["classes"] (the full
            # 20-class taxonomy used to drive the PCAP downloader; the
            # default output_data_file is the 10-class subset). Building
            # the model with the wrong count here would silently mismatch
            # a real 10-class checkpoint's fusion_head shape on load --
            # same bug as train_runner.py's, fixed there for the same
            # reason. Kept in this order (data before model) specifically
            # so num_classes is derived, not assumed.
            #
            # Defect 2 fix: benchmark on the held-out TEST split (the
            # manuscript's Table 2 says "500 test samples"), not
            # create_calibration_dataloader, which subsamples validation.
            test_loader = create_profiling_dataloader(num_samples=500)
            dummy_batch = next(iter(test_loader))
            # test_loader.dataset is Subset(Subset(TrafficDataset, test_idx),
            # profiling_idx) -- two levels of Subset wrapping, so three
            # `.dataset` hops to reach class_names.
            class_names = test_loader.dataset.dataset.dataset.class_names
            num_classes = len(class_names)

            # STEP B: Initialize Baseline Model
            orig_model = OptimizedTrafficCLIP(
                num_classes=num_classes,
                use_stats=args.use_stats,
                # Defect 2 fix: OptimizedTrafficCLIP.__init__ raises
                # ValueError when use_stats=True and stats_input_dim is
                # None -- this crashed before a tri-modal model could ever
                # be profiled.
                stats_input_dim=args.stats_input_dim if args.use_stats else None,
            ).to("cpu")

            # Load weights from a local checkpoint (see --checkpoint help).
            if args.checkpoint:
                checkpoint = torch.load(
                    args.checkpoint, map_location="cpu", weights_only=False
                )
                state_dict = (
                    checkpoint.state_dict()
                    if isinstance(checkpoint, torch.nn.Module)
                    else checkpoint
                )
                orig_model.load_state_dict(state_dict)
                logging.info(f"FP32 weights loaded from {args.checkpoint}.")
            else:
                logging.warning(
                    "No --checkpoint given; profiling a freshly-initialized "
                    "model. F1 will be near chance -- pass a trained "
                    "checkpoint for a meaningful accuracy comparison."
                )
            tokenizer = AutoTokenizer.from_pretrained(config["preprocess"]["tokenizer"])
            max_length = config["preprocess"]["max_length"]

            # STEP C: Export to ONNX
            # Defect 2 fix: export with the same use_stats setting the model
            # was built with, so the graph has 4 inputs (image, input_ids,
            # attention_mask, stats) when the tri-modal model is profiled --
            # exporting with the default use_stats=False silently dropped
            # the stats input regardless of args.use_stats.
            onnx_path = "traffic_clip_optimized.onnx"
            onnx_session = export_to_onnx(
                orig_model, dummy_batch, onnx_path, use_stats=args.use_stats
            )

            # Log the ONNX file as an artifact
            logging.info(f"Logging ONNX model saved at: {onnx_path}")
            mlflow.log_artifact(onnx_path)

            # STEP D: Benchmark
            evaluate_quantized_system(
                original_model=orig_model,
                optimized_model=onnx_session,  # Session instead of Module
                test_loader=test_loader,
                class_names=class_names,
                tokenizer=tokenizer,
                max_length=max_length,
                use_stats=args.use_stats,
            )

            mlflow.set_tag("status", "success")
            logging.info("ONNX Profiling successfully logged to local MLflow store.")

        except Exception as e:
            logging.error(f"Profiling Pipeline failed: {e}")
            import traceback

            logging.error(traceback.format_exc())
