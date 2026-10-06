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

from src.models.opt_traffic_clip import OptimizedTrafficCLIP
from src.optimization.calibration import create_calibration_dataloader

# from src.optimization.quant import quantize_traffic_model
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


# #     return quant_f1, quant_lat
# def get_model_size_mb(model_object):
#     """
#     Calculates the size of a model object in MB using cloudpickle.
#     """
#     # Serialize the model object to bytes
#     model_bytes = cloudpickle.dumps(model_object)

#     # Calculate size in Megabytes
#     size_mb = len(model_bytes) / (1024 * 1024)
#     return size_mb


# def evaluate_quantized_system(original_model, quantized_model, test_loader):
#     """
#     Enhanced evaluation merging accuracy testing with professional profiling.
#     Logs metrics directly to MLflow for side-by-side comparison.
#     """

#     def run_comprehensive_bench(model, loader, model_name):
#         model.eval()
#         model.to("cpu")

#         all_preds = []
#         all_labels = []
#         latencies = []

#         # 1. Warm-up Phase (for realistic CPU metrics)
#         warm_up_batches = 5
#         with torch.no_grad():
#             for i, batch in enumerate(loader):
#                 if i >= warm_up_batches:
#                     break
#                 img, stats, ids, mask = (
#                     batch["image"],
#                     batch["stats"],
#                     batch["input_ids"],
#                     batch["attention_mask"],
#                 )
#                 _ = model(img, ids, mask, stats)

#         # 2. Combined Accuracy & Latency Measurement
#         logger.info(f"Benchmarking {model_name}...")
#         with torch.no_grad():
#             for batch in loader:
#                 img = batch["image"]
#                 stats = batch["stats"]
#                 ids = batch["input_ids"]
#                 mask = batch["attention_mask"]
#                 labels = batch["label"]

#                 start = time.perf_counter()
#                 logits, _ = model(img, ids, mask, stats)
#                 end = time.perf_counter()

#                 batch_latency = ((end - start) * 1000) / img.size(0)
#                 latencies.append(batch_latency)

#                 preds = torch.argmax(logits, dim=1)
#                 all_preds.extend(preds.numpy())
#                 all_labels.extend(labels.numpy())

#         # 3. Calculate Metrics
#         f1 = f1_score(all_labels, all_preds, average="macro")
#         avg_lat = np.mean(latencies)
#         p99_lat = np.percentile(latencies, 99)
#         throughput = 1000 / avg_lat  # Flows per second

#         return f1, avg_lat, p99_lat, throughput

#     # Run benchmarks
#     orig_f1, orig_lat, orig_p99, orig_tp = run_comprehensive_bench(
#         original_model, test_loader, "Float32"
#     )
#     quant_f1, quant_lat, quant_p99, quant_tp = run_comprehensive_bench(
#         quantized_model, test_loader, "Int8"
#     )

#     # 4. Model Size Calculation
#     orig_size = get_model_size_mb(original_model)
#     quant_size = get_model_size_mb(quantized_model)
#     logging.info(f"Original Model Size: {orig_size:.2f} MB")
#     logging.info(f"Quantized Model Size: {quant_size:.2f} MB")

#     # --- MLFLOW logger ---
#     # Log metrics to MLflow
#     mlflow.log_metrics(
#         {
#             "orig_f1": orig_f1,
#             "quant_f1": quant_f1,
#             "orig_latency_ms": orig_lat,
#             "quant_latency_ms": quant_lat,
#             "orig_p99_ms": orig_p99,
#             "quant_p99_ms": quant_p99,
#             "orig_throughput_fps": orig_tp,
#             "quant_throughput_fps": quant_tp,
#             "compression_ratio": orig_size / quant_size,
#             "speedup_factor": orig_lat / quant_lat,
#         }
#     )

#     # --- logger REPORT ---
#     logger.info("\n" + "=" * 50)
#     logger.info(f"{'Metric':<20} | {'Original (FP32)':<12} | {'Quantized (INT8)':<12}")
#     logger.info(f"{'-'*20}-|-{'-'*12}-|-{'-'*12}")
#     logger.info(f"{'Size (MB)':<20} | {orig_size:<12.2f} | {quant_size:<12.2f}")
#     logger.info(f"{'Macro F1':<20} | {orig_f1:<12.4f} | {quant_f1:<12.4f}")
#     logger.info(f"{'Avg Lat (ms)':<20} | {orig_lat:<12.2f} | {quant_lat:<12.2f}")
#     logger.info(f"{'P99 Lat (ms)':<20} | {orig_p99:<12.2f} | {quant_p99:<12.2f}")
#     logger.info(f"{'Throughput':<20} | {orig_tp:<12.1f} | {quant_tp:<12.1f}")
#     logger.info(f"{'='*50}")
#     logger.info(
#         f"Summary: {orig_lat/quant_lat:.1f}x Speedup | {orig_size/quant_size:.1f}x Compression"
#     )


#     return quant_f1, quant_lat
def export_to_onnx(model, dummy_batch, save_path="model.onnx", use_stats=False):
    """
    Exports TrafficCLIP to ONNX with explicit input/output mapping.
    """
    import onnxruntime as ort

    logging.info(f"Exporting model to ONNX: {save_path}...")
    model.eval()
    model.to("cpu")

    img = dummy_batch["image"].cpu()
    ids = dummy_batch["input_ids"].cpu()
    mask = dummy_batch["attention_mask"].cpu()

    if use_stats:
        stats = dummy_batch["stats"].cpu()
        inputs = (img, ids, mask, stats)
        input_names = ["image", "input_ids", "attention_mask", "stats"]
        dynamic_axes = {
            "image": {0: "batch_size"},
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "stats": {0: "batch_size"},
        }
    else:
        inputs = (img, ids, mask)
        input_names = ["image", "input_ids", "attention_mask"]
        dynamic_axes = {
            "image": {0: "batch_size"},
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
        }

    torch.onnx.export(
        model,
        inputs,
        save_path,
        # Environment note, separate from the six defects: this repo's
        # torch (2.14) defaults torch.onnx.export to its newer dynamo-based
        # exporter, which fails tracing the in-place
        # `self.logit_scale.clamp_(0, ...)` inside a `no_grad()` block
        # (TrafficCLIP.forward / OptimizedTrafficCLIP.forward) with a
        # FakeTensor/Scalar overload mismatch. dynamo=False selects the
        # older TorchScript-based tracer, which handles this fine and is
        # almost certainly what produced the manuscript's Table 2 numbers
        # at the torch version those were run under.
        dynamo=False,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["logits", "scale"],
        dynamic_axes=dynamic_axes,
    )

    logging.info("ONNX Export successful.")

    session = ort.InferenceSession(
        save_path,
        providers=["CPUExecutionProvider"],
    )

    logging.info("ONNX Model Inputs:")
    for inp in session.get_inputs():
        logging.info(f" - {inp.name}")

    return session


# if __name__ == "__main__":
#     # --- 1. Path & Environment Setup ---
#     current_file = Path(__file__).resolve()
#     project_root = current_file.parents[2]

#     # Ensure project root is in path for custom module discovery
#     if str(project_root) not in sys.path:
#         sys.path.insert(0, str(project_root))
#     if str(project_root / "src") not in sys.path:
#         sys.path.insert(0, str(project_root / "src"))

#     # Register the model class for safe unpickling (PyTorch 2.6+)
#     import torch.serialization

#     from models.opt_traffic_clip import OptimizedTrafficCLIP

#     torch.serialization.add_safe_globals([OptimizedTrafficCLIP])

#     config = load_config()

#     parser = argparse.ArgumentParser(
#         description="Profile TrafficCLIP System Performance"
#     )
#     parser.add_argument("--lambda_cl", type=float, required=True)
#     parser.add_argument("--use_stats_prompts", action="store_true")
#     parser.add_argument("--model_version", type=str, default="optimized")
#     parser.add_argument("--use_stats", action="store_true", default=True)
#     parser.add_argument("--stats_input_dim", type=int, default=3)
#     args = parser.parse_args()

#     # Initialize DagsHub/MLflow
#     dagshub.init(
#         repo_owner=config["user"]["name"],
#         repo_name=config["user"]["ht_repo"],
#         mlflow=True,
#     )

#     logger = logging.getLogger("TrafficCLIP")  # Named logger
#     logger.setLevel(logging.INFO)

#     if not logger.handlers:
#         formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

#         # Stream (Terminal)
#         sh = logging.StreamHandler()
#         sh.setFormatter(formatter)
#         logger.addHandler(sh)

#         # File
#         fh = logging.FileHandler("profile.log")
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)

#     mlflow.set_experiment("TrafficCLIP_Quantization")
#     # --- 2. System Evaluation ---
#     with mlflow.start_run(run_name="System_Profiling_Run"):
#         try:
#             traffic_cfg = config["dataset"]["traffic"]["classes"]
#             num_classes = sum(len(c) for c in traffic_cfg.values())
#             exp_tag = f"{args.model_version}_L{args.lambda_cl}_stats{args.use_stats_prompts}_stats_data{args.use_stats}"

#             # STEP A: Setup Original FP32 Model
#             orig_model = OptimizedTrafficCLIP(
#                 num_classes=num_classes,
#                 use_stats=args.use_stats,
#                 stats_input_dim=args.stats_input_dim if args.use_stats else None,
#             ).to("cpu")

#             model_uri = f"models:/{exp_tag}/latest"
#             logging.info(f"Downloading FP32 weights from {model_uri}")
#             local_dir = mlflow.artifacts.download_artifacts(model_uri)

#             # Flexible Checkpoint Loading
#             weights_path = Path(local_dir) / "data" / "model.pth"
#             checkpoint = torch.load(
#                 weights_path, map_location="cpu", weights_only=False
#             )

#             # Extract state_dict if it's a model object or nested dict
#             if isinstance(checkpoint, torch.nn.Module):
#                 state_dict = checkpoint.state_dict()
#             elif isinstance(checkpoint, dict):
#                 state_dict = checkpoint.get(
#                     "model", checkpoint.get("state_dict", checkpoint)
#                 )
#             else:
#                 state_dict = checkpoint

#             orig_model.load_state_dict(state_dict)
#             logging.info("FP32 weights successfully mapped.")

#             # STEP B: Setup Quantized INT8 Model
#             logging.info("Building INT8 Quantized structure...")

#             # Start with a clean instance to perform the structural transform
#             dummy_model = OptimizedTrafficCLIP(
#                 num_classes=num_classes,
#                 use_stats=args.use_stats,
#                 stats_input_dim=args.stats_input_dim,
#             ).to("cpu")

#             # Transform structure (Fusion + Observers + Conversion)
#             try:
#                 quant_model = quantize_traffic_model(dummy_model)
#             except Exception as e:
#                 logging.error(f"Error occurred during quantization: {e}")
#                 raise e

#             # Load specific saved INT8 weights
#             int8_path = "quantized_traffic_model.pt"
#             if Path(int8_path).exists():
#                 # Using weights_only=False as quantized weights often contain metadata
#                 int8_checkpoint = torch.load(
#                     int8_path, map_location="cpu", weights_only=False
#                 )
#                 quant_model.load_state_dict(int8_checkpoint)
#                 logging.info(f"Successfully loaded INT8 weights from {int8_path}")
#             else:
#                 logging.warning(
#                     "Saved INT8 weights not found; profiling the dynamically converted instance."
#                 )

#         except Exception as e:
#             logging.error(f"Error during model setup: {e}")
#             raise e
