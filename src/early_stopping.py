import logging

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, mode="min"):
        """
        Args:
            patience (int): How many epochs to wait after last time improvement.
            delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, logs a message for each improvement.
            mode (str): 'min' for Loss, 'max' for Accuracy/F1 Score.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.mode = mode
        self.best_score = None
        self.no_improvement_count = 0
        self.stop_training = False

    def __call__(self, current_score):
        # Handle 'max' mode for Macro F1 or Accuracy
        if self.mode == "max":
            is_improvement = (self.best_score is None) or (
                current_score > self.best_score + self.delta
            )
        # Handle 'min' mode for Validation Loss
        else:
            is_improvement = (self.best_score is None) or (
                current_score < self.best_score - self.delta
            )

        if is_improvement:
            self.best_score = current_score
            self.no_improvement_count = 0
            if self.verbose:
                logging.info(
                    f"Improvement observed. New best score: {current_score:.4f}"
                )
        else:
            self.no_improvement_count += 1
            if self.verbose:
                logging.info(
                    f"No improvement for {self.no_improvement_count}/{self.patience} epochs."
                )

            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                logging.info("Stopping early to prevent overfitting.")
