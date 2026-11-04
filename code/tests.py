"""Smoke tests: python tests.py"""
import os, tempfile
import numpy as np
from run_stack import verify_tree
from datasets import load_kitti_tracks, to_scenarios
from sim import sample_normal, open_loop_gap
from stack import run_bt


def test_tree():
    n, props = verify_tree()
    assert n == 32 and all(v == n for v in props.values()), props


def test_oracle_bt_safe_on_everyday():
    P = sample_normal(np.random.default_rng(0), 300)
    sim, _ = run_bt(P, np.random.default_rng(0), mode="oracle")
    assert sim.collided.mean() == 0.0


def test_kitti_adapter():   # synthetic KITTI-format track: pedestrian walking at 1.5 m/s across the lane centre
    rows = [f"{i} 7 Pedestrian 0 0 0 0 0 0 0 1.7 0.6 0.8 {-3.0 + 1.5 * i / 10:.3f} 0 40.0 0" for i in range(40)]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f: f.write("\n".join(rows))
    s = to_scenarios(load_kitti_tracks(f.name), ego_speed=10.0); os.unlink(f.name)
    assert len(s) == 1 and abs(s[0]["ped_speed"] - 1.5) < 1e-3 and abs(s[0]["dist"] - 40.0) < 1e-6 and abs(s[0]["tau"] - (2.0 - 4.0)) < 0.2, s


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("ok", name)
