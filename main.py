"""Two-microphone SSL tutorial: simulate audio -> STFT -> GCC-PHAT -> TDOA -> DOA.

Usage examples:
    python main.py
    python main.py --angle 45
    python main.py --angle -30 --snr 10 --reverb

Conventions:
    Mic 1 is on the left (x=-d/2), Mic 2 on the right (x=+d/2).
    theta=0 is broadside, positive theta points right.
    TDOA = Mic 2 arrival time - Mic 1 arrival time.
    Thus positive theta means negative TDOA: theta=arcsin(-c*TDOA/d).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save PNG plots without requiring a GUI window
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, chirp, sosfiltfilt, stft

FS = 16000                   # Hz
C = 343.0                   # sound speed (m/s)
D = 0.20                    # microphone spacing (m)
DURATION = 2.0              # s
NFFT = 2048                 # FFT length
WINDOW_SIZE = 1024          # STFT window length
HOP_SIZE = 256              # STFT hop length
INTERP = 16                 # GCC interpolation factor


def signal_source(fs: int = FS, duration: float = DURATION, seed: int = 3) -> np.ndarray:
    """Generate a nonstationary synthetic broadband signal, not real speech."""
    rng = np.random.default_rng(seed)
    n = round(fs * duration)
    t = np.arange(n) / fs
    sos = butter(4, [250, 3400], btype="bandpass", fs=fs, output="sos")
    noise = sosfiltfilt(sos, rng.normal(size=n))
    sweep = chirp(t, f0=450, f1=2300, t1=duration, method="linear")
    # Apply a slowly varying amplitude envelope.
    envelope = (0.25 + 0.75 * (np.sin(2 * np.pi * 1.6 * t) ** 2))
    fade_n = min(n // 12, round(0.06 * fs))
    fade = np.ones(n)
    fade[:fade_n] = np.linspace(0, 1, fade_n)
    fade[-fade_n:] = np.linspace(1, 0, fade_n)
    s = (0.9 * noise + 0.25 * sweep) * envelope * fade
    return s / (np.max(np.abs(s)) + 1e-12)


def delay_by_samples(signal: np.ndarray, delay: float) -> np.ndarray:
    """Simulate fractional-sample delays by linear interpolation (educational approximation)."""
    indices = np.arange(signal.size, dtype=float)
    return np.interp(indices - delay, indices, signal, left=0.0, right=0.0)


def simulate_pair(
    angle_deg: float,
    source: np.ndarray,
    *,
    fs: int = FS,
    mic_distance: float = D,
    speed: float = C,
    snr_db: float = 30.0,
    reverb: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Simulate a microphone pair; return (Mic1, Mic2, true TDOA in seconds)."""
    if not -90 <= angle_deg <= 90:
        raise ValueError("This two-microphone example supports only -90 to +90 degrees.")
    rng = np.random.default_rng(seed)
    true_tau21 = -(mic_distance / speed) * np.sin(np.deg2rad(angle_deg))
    base_delay = 0.035 * fs
    delay1 = base_delay - true_tau21 * fs / 2
    delay2 = base_delay + true_tau21 * fs / 2
    mic1 = delay_by_samples(source, delay1)
    mic2 = delay_by_samples(source, delay2)

    if reverb:
        # Toy reflection paths; NOT a calibrated physical room impulse response.
        mic1 = mic1 + 0.50 * delay_by_samples(source, delay1 + 0.006 * fs)
        mic1 = mic1 + 0.35 * delay_by_samples(source, delay1 + 0.014 * fs)
        mic2 = mic2 + 0.56 * delay_by_samples(source, delay2 + 0.008 * fs)
        mic2 = mic2 + 0.32 * delay_by_samples(source, delay2 + 0.017 * fs)

    for x in (mic1, mic2):
        p_signal = np.mean(x * x)
        p_noise = p_signal / (10 ** (snr_db / 10))
        x += rng.normal(0.0, np.sqrt(p_noise), size=x.shape)
    return mic1, mic2, true_tau21


def gcc_phat_stft(
    mic1: np.ndarray,
    mic2: np.ndarray,
    *,
    fs: int = FS,
    distance: float = D,
    speed: float = C,
    nfft: int = NFFT,
    window: int = WINDOW_SIZE,
    hop: int = HOP_SIZE,
    interp: int = INTERP,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Average STFT cross-spectrum, PHAT weight, IFFT, and physical-range peak."""
    freqs, times, z1 = stft(mic1, fs=fs, window="hann", nperseg=window,
                            noverlap=window - hop, nfft=nfft,
                            boundary=None, padded=False)
    _, _, z2 = stft(mic2, fs=fs, window="hann", nperseg=window,
                    noverlap=window - hop, nfft=nfft,
                    boundary=None, padded=False)

    # The Mic2 * conj(Mic1) peak corresponds to tau21=t2-t1; channel order matters.
    cross_spectrum = np.mean(z2 * np.conj(z1), axis=1)
    phat = cross_spectrum / (np.abs(cross_spectrum) + 1e-12)
    correlation = np.fft.irfft(phat, n=nfft * interp)

    # Search the physically feasible lag range plus one sample of margin.
    max_lag = min(int(np.ceil(distance / speed * fs * interp)) + interp,
                  len(correlation) // 2 - 1)
    lags = np.arange(-max_lag, max_lag + 1)
    curve = np.concatenate((correlation[-max_lag:], correlation[:max_lag + 1]))
    peak_lag = lags[np.argmax(curve)] / interp
    tau21 = peak_lag / fs
    return tau21, lags / interp, curve, freqs, times, z1


def tau_to_angle(tau21: float, distance: float = D, speed: float = C) -> float:
    ratio = -speed * tau21 / distance
    return float(np.rad2deg(np.arcsin(np.clip(ratio, -1, 1))))


def save_plots(out_dir: Path, mic1: np.ndarray, mic2: np.ndarray,
               lags: np.ndarray, curve: np.ndarray, peak_tau: float,
               freqs: np.ndarray, times: np.ndarray, stft_mic1: np.ndarray,
               fs: int = FS) -> None:
    t = np.arange(mic1.size) / fs
    region = (t >= 0.06) & (t <= 0.074)
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t[region] * 1000, mic1[region], label="Mic 1 (left)", linewidth=1)
    ax.plot(t[region] * 1000, mic2[region], label="Mic 2 (right)", linewidth=1, alpha=0.75)
    ax.set(xlabel="Time (ms)", ylabel="Amplitude", title="Two microphone waveforms (short excerpt)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "01_waveforms.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    db = 20 * np.log10(np.abs(stft_mic1) + 1e-8)
    mesh = ax.pcolormesh(times, freqs, db, shading="auto", cmap="viridis")
    ax.set(xlabel="Time (s)", ylabel="Frequency (Hz)", ylim=(0, 4000),
           title="Mic 1 STFT magnitude spectrogram")
    fig.colorbar(mesh, ax=ax, label="Magnitude (dB, relative)")
    fig.tight_layout()
    fig.savefig(out_dir / "02_stft.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(lags, curve)
    ax.axvline(peak_tau * fs, linestyle="--", label=f"Estimated lag: {peak_tau * fs:.2f} samples")
    ax.set(xlabel="Lag (samples, Mic2 relative to Mic1)", ylabel="GCC-PHAT correlation",
           title="GCC-PHAT peak gives TDOA")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "03_gcc_phat.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-microphone SSL: STFT -> GCC-PHAT -> TDOA -> DOA")
    parser.add_argument("--angle", type=float, default=30, help="True source angle in degrees; positive is to the right")
    parser.add_argument("--snr", type=float, default=30, help="Simulated signal-to-noise ratio in dB")
    parser.add_argument("--reverb", action="store_true", help="Enable simplified multipath reflections")
    parser.add_argument("--out", type=str, default="outputs", help="Directory for output files")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = signal_source()
    mic1, mic2, truth_tau = simulate_pair(args.angle, source, snr_db=args.snr,
                                          reverb=args.reverb, seed=42)
    pred_tau, lag_samples, corr, freqs, times, z1 = gcc_phat_stft(mic1, mic2)
    pred_angle = tau_to_angle(pred_tau)
    error = pred_angle - args.angle
    print("======== Two-Microphone Sound Source Localization ========")
    print(f"Angle convention: broadside=0 deg, right=positive; Mic1=left, Mic2=right; d={D:.2f} m")
    print(f"True DOA              : {args.angle:+.2f} deg")
    print(f"True TDOA (t2-t1)     : {truth_tau * 1e6:+.2f} us = {truth_tau * FS:+.3f} samples")
    print(f"Estimated TDOA (t2-t1): {pred_tau * 1e6:+.2f} us = {pred_tau * FS:+.3f} samples")
    print(f"Estimated DOA         : {pred_angle:+.2f} deg")
    print(f"Absolute angular error: {abs(error):.2f} deg")
    print(f"SNR                   : {args.snr:.1f} dB; Simplified reflections: {args.reverb}")
    print(f"Output directory      : {out_dir.resolve()}")

    wav_stereo = np.column_stack([mic1, mic2])
    wav_stereo /= np.max(np.abs(wav_stereo)) + 1e-12
    wavfile.write(out_dir / "two_mics.wav", FS, (wav_stereo * 32767).astype(np.int16))
    save_plots(out_dir, mic1, mic2, lag_samples, corr, pred_tau, freqs, times, z1)
    with (out_dir / "result.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["angle_true_deg", "tdoa_true_us", "tdoa_est_us",
                         "angle_est_deg", "absolute_error_deg", "snr_db", "reverb"])
        writer.writerow([args.angle, truth_tau * 1e6, pred_tau * 1e6,
                         pred_angle, abs(error), args.snr, args.reverb])


if __name__ == "__main__":
    main()
