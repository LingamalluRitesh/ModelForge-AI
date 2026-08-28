"""
ModelForge AI - ML Engine: Audio & Signal DSP Feature Extraction
Implements Short-Time Fourier Transform (STFT), Mel-Scale Filterbanks,
Mel-Frequency Cepstral Coefficients (MFCC), Spectral Centroid, and Spectral Roll-off.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np


class AudioFeatureExtractor:
    """Standard DSP audio spectral analysis engine implemented in NumPy."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 256,
        n_mels: int = 40,
        n_mfcc: int = 13,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def stft(self, signal: np.ndarray) -> np.ndarray:
        """Compute Short-Time Fourier Transform with Hanning window."""
        window = np.hanning(self.n_fft)
        n_frames = 1 + (len(signal) - self.n_fft) // self.hop_length

        if n_frames <= 0:
            return np.zeros((self.n_fft // 2 + 1, 1), dtype=np.complex128)

        stft_matrix = np.empty((self.n_fft // 2 + 1, n_frames), dtype=np.complex128)

        for i in range(n_frames):
            start = i * self.hop_length
            frame = signal[start : start + self.n_fft] * window
            stft_matrix[:, i] = np.fft.rfft(frame)

        return stft_matrix

    def _hz_to_mel(self, hz: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(self, mel: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def mel_filterbank(self) -> np.ndarray:
        """Construct triangular Mel-scale filterbank matrix."""
        low_freq = 0.0
        high_freq = self.sample_rate / 2.0

        low_mel = self._hz_to_mel(low_freq)
        high_mel = self._hz_to_mel(high_freq)
        mel_points = np.linspace(low_mel, high_mel, self.n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)

        bin_points = np.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        fbank = np.zeros((self.n_mels, self.n_fft // 2 + 1))

        for m in range(1, self.n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin_points[m - 1]) / max(1, (bin_points[m] - bin_points[m - 1]))
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin_points[m + 1] - k) / max(1, (bin_points[m + 1] - bin_points[m]))

        return fbank

    def mfcc(self, signal: np.ndarray) -> np.ndarray:
        """Compute Mel-Frequency Cepstral Coefficients (MFCC)."""
        stft_res = self.stft(signal)
        power_spectrum = (np.abs(stft_res) ** 2) / self.n_fft

        fbank = self.mel_filterbank()
        mel_spectrum = np.dot(fbank, power_spectrum)
        mel_spectrum = np.maximum(mel_spectrum, 1e-10)
        log_mel = np.log(mel_spectrum)

        # Type-II Discrete Cosine Transform (DCT)
        n_mels, n_frames = log_mel.shape
        mfcc_out = np.zeros((self.n_mfcc, n_frames))

        for k in range(self.n_mfcc):
            basis = np.cos(np.pi * k * (np.arange(n_mels) + 0.5) / n_mels)
            mfcc_out[k, :] = np.dot(basis, log_mel)

        return mfcc_out

    def spectral_centroid(self, signal: np.ndarray) -> np.ndarray:
        """Calculate center of mass of power spectrum over time frames."""
        stft_res = np.abs(self.stft(signal))
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)

        num = np.dot(freqs, stft_res)
        den = np.sum(stft_res, axis=0)
        den = np.maximum(den, 1e-10)

        return num / den
