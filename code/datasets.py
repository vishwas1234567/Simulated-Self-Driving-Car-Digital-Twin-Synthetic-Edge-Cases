"""Convert real pedestrian tracks into scenario vectors z = (v0, dist, ped_speed, tau, occl, ped_r, dropout, noise).
Supported input: KITTI tracking label files (camera frame: x right, z forward, 10 Hz).
Usage: python datasets.py <label_file.txt> <ego_speed_m_s>"""
import sys
import numpy as np

FPS = 10.0


def load_kitti_tracks(path):
    tracks = {}
    for ln in open(path):
        f = ln.split()
        if len(f) < 17 or f[2] != "Pedestrian": continue
        frame, tid = int(f[0]), int(f[1])
        tracks.setdefault(tid, []).append((frame / FPS, float(f[13]), float(f[15]), float(f[11])))  # t, x, z, width
    return tracks


def to_scenarios(tracks, ego_speed, min_len=8):
    """Pedestrians whose lateral position changes sign (they cross the ego lane centre line)."""
    out = []
    for tid, rows in tracks.items():
        t, x, z, w = np.array(sorted(rows)).T
        if len(t) < min_len or x.min() * x.max() >= 0: continue
        k, b = np.polyfit(t, x, 1)                      # x(t) = k t + b
        tc = -b / k                                      # time the pedestrian crosses the lane centre
        zc = float(np.interp(tc, t, z))
        out.append(dict(track=tid, v0=ego_speed, dist=zc, ped_speed=abs(k), tau=tc - zc / max(ego_speed, 0.1),
                        occl=0.0, ped_r=float(np.clip(np.median(w) / 2, 0.25, 1.0)), dropout=0.0, noise=0.05))
    return out


if __name__ == "__main__":
    for s in to_scenarios(load_kitti_tracks(sys.argv[1]), float(sys.argv[2])): print(s)
