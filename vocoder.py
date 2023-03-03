import argparse
from typing import Optional
import numpy as np
import librosa
import soundfile


def phase_vocoder(
    D: np.array,
    rate: float,
    hop_length: Optional[int] = None,
    n_fft: Optional[int] = None,
    ) -> np.array:

    n_fft = n_fft or 2 * (D.shape[-2] - 1)
    hop_length = hop_length or int(n_fft // 4)

    time_steps = np.arange(0, D.shape[-1], rate, dtype=np.float64)

    shape = list(D.shape)
    shape[-1] = len(time_steps)
    d_stretch = np.zeros(shape=shape, dtype=D.dtype)

    phi_advance = np.linspace(0, np.pi * hop_length, D.shape[-2])

    phase_acc = np.angle(D[:, 0])

    padding = [(0, 0) for _ in D.shape]
    padding[-1] = (0, 2)
    D = np.pad(D, padding, mode="constant")

    for t, step in enumerate(time_steps):
        start = int(step)
        end = start + 2
        columns = D[:, start:end]

        alpha = step - start

        mag = (1.0 - alpha) * np.abs(columns[:, 0]) + alpha * np.abs(columns[:, 1])

        d_stretch[:, t] = np.cos(phase_acc) + 1j * np.sin(phase_acc) * mag

        dphase = np.angle(columns[:, 1]) - np.angle(columns[:, 0]) - phi_advance
        dphase = np.mod(dphase + np.pi, 2 * np.pi) - np.pi

        phase_acc += phi_advance + dphase

    return d_stretch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase Vocoder")
    parser.add_argument("input", type=str, help="input wav file")
    parser.add_argument("output", type=str, help="output wav file")
    parser.add_argument("time_stretch_ratio", type=int, help="time stretch ratio")
    args = parser.parse_args()
    y, sr = librosa.load(args.input)
    D = librosa.stft(y, n_fft=2048, hop_length=512)
    D_fast = phase_vocoder(D, rate=args.time_stretch_ratio, hop_length=512)
    y_fast = librosa.istft(D_fast, hop_length=512)
    soundfile.write(args.output, y_fast, sr)
