"""
Tests for A/train.py's loss and scheduling helpers, isolated from the model
itself so they run in well under a second.
"""

import torch

from A.train import build_scheduler, supervised_contrastive_loss


def test_supervised_contrastive_loss_is_finite_with_repeated_classes():
    torch.manual_seed(0)
    v_e = torch.nn.functional.normalize(torch.randn(8, 16), dim=-1)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supervised_contrastive_loss(v_e, labels)
    assert torch.isfinite(loss)
    assert loss.item() >= 0


def test_supervised_contrastive_loss_handles_all_singleton_classes():
    # Every class is a singleton in this batch, so there are no positives at
    # all. This used to return NaN, because log_prob * positive_mask
    # multiplies the self-masked -inf diagonal by a 0/1 mask.
    torch.manual_seed(0)
    v_e = torch.nn.functional.normalize(torch.randn(5, 16), dim=-1)
    labels = torch.tensor([0, 1, 2, 3, 4])
    loss = supervised_contrastive_loss(v_e, labels)
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_supervised_contrastive_loss_mixed_singleton_and_grouped():
    torch.manual_seed(1)
    v_e = torch.nn.functional.normalize(torch.randn(6, 16), dim=-1)
    labels = torch.tensor([0, 0, 1, 2, 3, 3])
    loss = supervised_contrastive_loss(v_e, labels)
    assert torch.isfinite(loss)


def test_scheduler_constant_during_warmup_then_decays_to_near_zero():
    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.002, momentum=0.9)
    steps_per_epoch = 10
    scheduler = build_scheduler(
        optimizer, base_lr=0.002, warmup_lr=1e-5, warmup_epochs=1,
        total_epochs=5, steps_per_epoch=steps_per_epoch,
    )

    lrs = []
    for _ in range(5 * steps_per_epoch):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    # Warm-up epoch: constant at warmup_lr.
    assert all(abs(lr - 1e-5) < 1e-9 for lr in lrs[:steps_per_epoch])
    # Post-warmup: starts at base_lr and decays monotonically to near 0.
    assert abs(lrs[steps_per_epoch] - 0.002) < 1e-9
    assert lrs[-1] < lrs[steps_per_epoch]
