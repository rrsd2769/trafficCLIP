import logging
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F

from src.dataset import get_dataloader


class TrafficGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Define hooks to capture gradients and activations
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_image, input_ids, attention_mask, target_class):
        self.model.eval()

        # FIX: Enable gradients for the input image to allow backprop to the Conv layer
        input_image.requires_grad = True

        # Forward pass through the Optimized architecture
        logits, _ = self.model(input_image, input_ids, attention_mask)

        # Zero gradients and backpropagate for the target class
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        # Global average pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Weight the activations by the gradients
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Create heatmap via ReLU on weighted sum
        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)

        # Normalize for visualization
        heatmap /= torch.max(heatmap)
        return heatmap.detach().cpu().numpy()


def debug_misclassifications(
    model, args, config, device, experiment_dir, target_conflicts
):
    """
    Specifically targets Gmail/BitTorrent and Gmail/Skype conflicts.
    """

    NPZ_PATH = config["paths"]["output_data_file"]
    TOKENIZER_NAME = config["preprocess"]["tokenizer"]
    MAX_LENGTH = config["test"]["max_length"]
    BATCH_SIZE = config["test"]["batch_size"]
    use_dynamic_prompts = args.use_stats_prompts
    current_seed = args.seed

    _, _, test_loader = get_dataloader(
        npz_path=NPZ_PATH,
        tokenizer=TOKENIZER_NAME,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
        seed=current_seed,
        use_dynamic_prompts=use_dynamic_prompts,
    )
    # Target the first Conv layer of the Detail Encoder for raw byte patterns
    cam = TrafficGradCAM(model, model.detail_encoder.initial_conv[0])

    model.eval()
    save_path = Path(experiment_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    class_names = test_loader.dataset.dataset.class_names
    # count = 0
    all_figs = []

    max_search_batches = 50
    max_figs_to_collect = 10

    logging.info(f"Starting misclassification search for: {target_conflicts}")

    # 3. Search Loop
    for batch_idx, batch in enumerate(test_loader):
        if len(all_figs) >= max_figs_to_collect or batch_idx > max_search_batches:
            break

        images = batch["image"].to(device)
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        stats = batch["stats"].to(device)

        # Forward pass without gradient tracking
        with torch.no_grad():
            if args.model_version == "optimized":
                if args.use_stats:
                    logits, _ = model(images, ids, mask, stats)
                else:
                    logits, _ = model(images, ids, mask)
            else:
                logits, _ = model(images, ids, mask)

        preds = torch.argmax(logits, dim=1)

        # Check each sample in the batch
        for i in range(len(labels)):
            true_label = class_names[labels[i].item()]
            pred_label = class_names[preds[i].item()]

            # Dynamic Conflict Filter: Only process if it's a conflict we care about
            if (true_label, pred_label) in target_conflicts:
                logging.info(
                    f"Targeted conflict found: {true_label} predicted as {pred_label}"
                )

                # Generate Heatmap (Grad-CAM handles its own internal gradients)
                heatmap = cam.generate_heatmap(
                    images[i : i + 1], ids[i : i + 1], mask[i : i + 1], preds[i].item()
                )

                # 4. Plotting
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))

                # Original Traffic Image
                ax[0].imshow(images[i].cpu().squeeze(), cmap="gray")
                ax[0].set_title(f"True: {true_label}")
                ax[0].axis("off")

                # Grad-CAM Heatmap
                ax[1].imshow(heatmap, cmap="jet")
                ax[1].set_title(f"Pred: {pred_label} (Heatmap)")
                ax[1].axis("off")

                # Save locally for safety
                file_name = f"debug_{len(all_figs)}_{true_label}_to_{pred_label}.png"
                img_path = save_path / file_name
                plt.savefig(img_path, bbox_inches="tight")

                # Add to return list
                all_figs.append(fig)

                if len(all_figs) >= max_figs_to_collect:
                    break

    if not all_figs:
        logging.warning(
            "Search complete: No targeted conflicts found in the provided batches."
        )

    return all_figs
