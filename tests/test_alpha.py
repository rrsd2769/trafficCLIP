"""
Regression test for the adapter blend weight.

`OptimizedTrafficCLIP.__init__` called `super().__init__(num_classes)`, but
`TrafficCLIP.__init__` takes `alpha`, not `num_classes`, so every "optimized"
run trained with alpha=10 (10*Adapter(f2) - 9*f2) rather than the
manuscript's Eq. 3 value of 0.9.
"""

from src.models.opt_traffic_clip import OptimizedTrafficCLIP


def test_default_alpha_is_point_nine():
    model = OptimizedTrafficCLIP(num_classes=10, use_stats=False)
    assert model.alpha == 0.9


def test_alpha_still_overridable_for_the_pre_fix_report():
    # alpha=10 is what produced the manuscript's Table 1, so it has to stay
    # reachable on purpose in order to report both values side by side.
    model = OptimizedTrafficCLIP(num_classes=10, use_stats=False, alpha=10)
    assert model.alpha == 10
