"""Lesson 06-7: Evaluate IPD network vs GCC-PHAT under noise/reflections.

Prerequisite: Run lesson06_6.py once to create outputs_ipd/ipd_doa_model.pth.
Run: .\\.venv\\Scripts\\python.exe lesson06_7.py --count 50
All console output, CSV labels and figures are in English.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.signal import butter
from torch import nn

from main import FS, C, D, simulate_pair, gcc_phat_stft, tau_to_angle
from lesson06_6 import create_source, extract_ipd

ROOT = Path(__file__).resolve().parent
SNR_VALUES = [30, 20, 10, 0, -10]


def load_model(path: Path) -> nn.Module:
    if not path.is_file():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "First run: .\\.venv\\Scripts\\python.exe lesson06_6.py"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["sample_rate"] != FS or not np.isclose(checkpoint["mic_spacing_m"], D):
        raise ValueError("Model geometry/sample rate does not match main.py")
    feature_count = int(checkpoint["feature_count"])
    model = nn.Sequential(
        nn.Linear(feature_count, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1), nn.Tanh(),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def evaluate(model: nn.Module, count: int, seed: int, output: Path) -> None:
    rng = np.random.default_rng(seed)
    sos = butter(4, [120, 3000], btype="bandpass", fs=FS, output="sos")

    # Generate one fresh set of held-out source/angle pairs.
    # Reuse these exact pairs across conditions for controlled comparisons.
    examples = []
    for _ in range(count):
        angle = float(rng.uniform(-70, 70))
        source = create_source(rng, sos)
        noise_seed = int(rng.integers(0, 2**31))
        examples.append((angle, source, noise_seed))

    output.mkdir(parents=True, exist_ok=True)
    details, summary = [], []
    print(f"Evaluation examples per condition: {count}")
    print("Training conditions: direct path, random SNR 8-30 dB")
    print("Test conditions: SNR sweep with and without simplified reflections")

    for reflection in (False, True):
        for snr in SNR_VALUES:
            truth = []
            features = []
            gcc_preds = []
            for angle, source, noise_seed in examples:
                mic1, mic2, _ = simulate_pair(
                    angle, source, snr_db=snr,
                    reverb=reflection, seed=noise_seed,
                )
                features.append(extract_ipd(mic1, mic2))
                tdoa, *_ = gcc_phat_stft(mic1, mic2, fs=FS)
                gcc_preds.append(tau_to_angle(tdoa))
                truth.append(angle)

            features = np.stack(features)
            truth = np.asarray(truth)
            gcc_preds = np.asarray(gcc_preds)
            with torch.inference_mode():
                nn_sin = model(torch.from_numpy(features)).numpy().ravel()
            nn_preds = np.rad2deg(np.arcsin(np.clip(nn_sin, -1, 1)))
            nn_errors = np.abs(nn_preds - truth)
            gcc_errors = np.abs(gcc_preds - truth)

            row = {
                "reverb": reflection,
                "snr_db": snr,
                "n": count,
                "nn_mae_deg": float(np.mean(nn_errors)),
                "gcc_mae_deg": float(np.mean(gcc_errors)),
                "nn_rmse_deg": float(np.sqrt(np.mean((nn_preds - truth) ** 2))),
                "gcc_rmse_deg": float(np.sqrt(np.mean((gcc_preds - truth) ** 2))),
            }
            summary.append(row)
            print(
                f"Reverb={str(reflection):5s} | SNR={snr:>3} dB | "
                f"NN MAE={row['nn_mae_deg']:6.2f} deg | "
                f"GCC MAE={row['gcc_mae_deg']:6.2f} deg"
            )
            for i in range(count):
                details.append({
                    "reverb": reflection,
                    "snr_db": snr,
                    "sample_index": i,
                    "true_doa_deg": float(truth[i]),
                    "nn_doa_deg": float(nn_preds[i]),
                    "gcc_doa_deg": float(gcc_preds[i]),
                    "nn_abs_error_deg": float(nn_errors[i]),
                    "gcc_abs_error_deg": float(gcc_errors[i]),
                })

    for filename, rows in (("summary.csv", summary), ("individual_results.csv", details)):
        with (output / filename).open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for reflection, linestyle in ((False, "-"), (True, "--")):
        subset = sorted((r for r in summary if r["reverb"] == reflection),
                        key=lambda r: r["snr_db"])
        x = [r["snr_db"] for r in subset]
        for alg, marker in (("nn", "o"), ("gcc", "s")):
            label = f"{'IPD network' if alg == 'nn' else 'GCC-PHAT'}, reverb={reflection}"
            ax.plot(x, [r[f"{alg}_mae_deg"] for r in subset],
                    marker=marker, linestyle=linestyle, label=label)
    ax.set(xlabel="SNR (dB)", ylabel="Mean absolute error (deg)",
           title="SSL generalization under noise and simplified reflections")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "mae_vs_snr.png", dpi=160)
    plt.close(fig)
    print(f"Saved CSV and plot in: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SSL generalization")
    parser.add_argument("--count", type=int, default=50,
                        help="Independent examples per condition (default: 50)")
    parser.add_argument("--seed", type=int, default=50001,
                        help="Independent evaluation seed")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs_generalization")
    args = parser.parse_args()
    if args.count < 2:
        parser.error("--count must be >= 2")
    torch.set_num_threads(1)
    model = load_model(ROOT / "outputs_ipd" / "ipd_doa_model.pth")
    evaluate(model, args.count, args.seed, args.out)


if __name__ == "__main__":
    main()
