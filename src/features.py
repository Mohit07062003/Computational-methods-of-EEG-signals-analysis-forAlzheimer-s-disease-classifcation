import numpy as np
import pandas as pd
import pywt
import antropy as ant
from typing import List, Dict, Any

from .preprocessing import FS, WAVELET_NAME, DECOMPOSITION_LEVEL, FREQUENCY_BANDS, BAND_LEVEL_MAP, decompose_signal, reconstruct_band_signal

QE_M = 2
QE_R = 0.15

QG_NQG = 20
QG_K = 1

VG_MIN_LENGTH = 3


def calculate_wavelet_coherence(signal1: np.ndarray, signal2: np.ndarray, fs: int = FS, wavelet: str = WAVELET_NAME) -> float | None:
    if signal1 is None or signal2 is None or len(signal1) != len(signal2) or len(signal1) == 0:
        return None

    signal1 = signal1.astype(float)
    signal2 = signal2.astype(float)

    if np.all(signal1 == signal1[0]) or np.all(signal2 == signal2[0]) or np.any(np.isnan(signal1)) or np.any(np.isnan(signal2)) or np.any(np.isinf(signal1)) or np.any(np.isinf(signal2)):
         return None

    try:
        n = len(signal1)
        if n <= 1: return None

        scales = np.arange(1, int(n/2) + 1)
        if len(scales) < 2:
            return None

        coherence, _, freq_output, _, _, _ = pywt.wct(signal1, signal2, wavelet, scales=scales, dt=1.0/fs)

        min_freq, max_freq = 0.5, 30.0
        relevant_freq_indices = np.where((freq_output >= min_freq) & (freq_output <= max_freq))[0]

        if len(relevant_freq_indices) == 0:
             return None

        relevant_coherence = coherence[relevant_freq_indices, :]
        avg_coherence = np.nanmean(relevant_coherence)

        if np.isnan(avg_coherence):
             return None

        return float(avg_coherence)

    except Exception:
        return None


def katz_fd(signal: np.ndarray) -> float | None:
    if signal is None or len(signal) <= 1:
        return None

    signal = signal.astype(float)
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return None

    n = len(signal)

    if np.all(signal == signal[0]):
        return None

    try:
        fd = ant.katz_fd(signal)
        if np.isinf(fd) or np.isnan(fd):
             return None
        return float(fd)
    except Exception:
        return None


def calculate_quadratic_entropy(signal: np.ndarray, m: int = QE_M, r: float = QE_R) -> float | None:
    if signal is None or len(signal) <= m:
        return None

    signal = signal.astype(float)
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return None

    sd = np.std(signal)
    if sd < 1e-9:
         return None

    tolerance = r * sd

    if tolerance <= 0:
        return None

    try:
        sampen = ant.sample_entropy(signal, order=m, r=tolerance)

        if np.isinf(sampen) or np.isnan(sampen):
             return None

        if r <= 0:
             return None
        log_term = np.log(2 * r) if 2 * r > 0 else -np.inf

        if not np.isfinite(log_term):
             return None

        quadratic_entropy = sampen + log_term

        if not np.isfinite(quadratic_entropy):
             return None

        return float(quadratic_entropy)
    except Exception:
        return None


def calculate_relative_wavelet_energy(coeffs: List[np.ndarray] | None, band_level_map: Dict[str, List[int]] = BAND_LEVEL_MAP, frequency_bands: List[str] = FREQUENCY_BANDS) -> Dict[str, float]:
    relative_energies = {}
    if coeffs is None or not isinstance(coeffs, list) or len(coeffs) == 0:
        return {}

    try:
        total_energy = 0
        for c in coeffs:
            if c is not None:
                 total_energy += np.sum(np.square(c))

        if total_energy < 1e-12:
             return {}

        for band in frequency_bands:
            if band in band_level_map:
                band_energy = 0
                for level_idx in band_level_map[band]:
                     if level_idx < len(coeffs) and coeffs[level_idx] is not None:
                        band_energy += np.sum(np.square(coeffs[level_idx]))

                if band_energy < 0:
                     continue

                relative_energies[band] = band_energy / total_energy

        return relative_energies
    except Exception:
        return {}


def calculate_quantile_graph_mjl(signal: np.ndarray, nqg: int = QG_NQG, k: int = QG_K) -> float | None:
    if signal is None or len(signal) <= k or len(signal) < nqg:
        return None

    signal = signal.astype(float)
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return None

    n = len(signal)

    try:
        try:
             quantile_indices, bins_edges = pd.qcut(signal, q=nqg, labels=False, retbins=True, duplicates='drop')

             if np.any(np.isnan(quantile_indices)):
                 return None

             actual_nqg = len(bins_edges) - 1

             if actual_nqg < 2:
                 return None

             quantile_indices = quantile_indices.astype(int)

        except ValueError:
             return None
        except Exception:
            return None

        Wk = np.zeros((actual_nqg, actual_nqg))
        valid_transitions = 0
        max_idx = actual_nqg - 1
        for i in range(n - k):
            q_from = quantile_indices[i]
            q_to = quantile_indices[i + k]
            if 0 <= q_from <= max_idx and 0 <= q_to <= max_idx:
                 Wk[q_from, q_to] += 1
                 valid_transitions += 1

        if valid_transitions == 0:
             return None

        P = np.abs(np.arange(actual_nqg)[:, np.newaxis] - np.arange(actual_nqg))

        try:
             trace_product = np.trace(np.dot(Wk.T, P))
             mean_jump_length = trace_product / actual_nqg
        except Exception:
             return None

        if not np.isfinite(mean_jump_length):
             return None

        return float(mean_jump_length)

    except Exception:
        return None


def check_obscured(k: int, j: int, i: int, signal: np.ndarray) -> bool:
    if not (0 <= k < j < i < len(signal)):
        return False

    val1 = np.float64(signal[j] - signal[k]) * (i - k)
    val2 = np.float64(signal[i] - signal[k]) * (j - k)
    return val1 <= val2


def build_standard_vg_adj_matrix_fast(signal: np.ndarray) -> np.ndarray | None:
    n = len(signal)
    if n < VG_MIN_LENGTH:
        return None
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return None

    adj_matrix = np.zeros((n, n), dtype=int)
    stack = [] # Stack stores indices

    try:
        for i in range(n):
            while stack:
                j = stack[-1]

                is_obscured_by_stack = False
                if len(stack) >= 2:
                    k = stack[-2]
                    if check_obscured(k, j, i, signal):
                        is_obscured_by_stack = True

                if is_obscured_by_stack:
                    stack.pop()
                else:
                    adj_matrix[j, i] = 1
                    adj_matrix[i, j] = 1
                    break

            stack.append(i)

        return adj_matrix

    except Exception:
        return None


def calculate_visibility_graph_iota(signal: np.ndarray, fs: int = FS) -> float | None:
    if signal is None or len(signal) < VG_MIN_LENGTH:
        return None
    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
        return None

    try:
        adj_matrix = build_standard_vg_adj_matrix_fast(signal)

        if adj_matrix is None or adj_matrix.shape[0] != len(signal):
             return None

        try:
             eigenvalues = np.linalg.eigvals(adj_matrix)
             lambda_max = np.max(np.abs(eigenvalues))
        except np.linalg.LinAlgError:
             return None
        except Exception:
             return None

        if not np.isfinite(lambda_max) or lambda_max < 0:
             return None

        N = len(signal)
        pi_term = np.pi / (N + 1)
        cos_pi_term = np.cos(pi_term)

        numerator_c = lambda_max - 2 * cos_pi_term
        denominator_c_paper = (N - 1) - 2 * cos_pi_term

        if abs(denominator_c_paper) < 1e-9:
             return None

        c = numerator_c / denominator_c_paper

        iota = 4 * c * (1 - c)

        if not np.isfinite(iota):
             return None

        return float(iota)
    except Exception:
        return None


def extract_all_features_for_subject(patient_info: Dict[str, Any], channels: List[str], fs: int, expected_signal_length: int) -> Dict[str, Any]:
    subject_features = {
        'Subject': patient_info['name'],
        'Group': patient_info['Group'],
        'Condition': patient_info['Condition']
    }

    valid_channels_data = {ch: sig for ch, sig in patient_info['data'].items()
                           if sig is not None and len(sig) == expected_signal_length
                           and not np.any(np.isnan(sig)) and not np.any(np.isinf(sig))}

    for channel, signal in valid_channels_data.items():
        current_original_signal_len = len(signal)

        features_F_orig = katz_fd(signal)
        if features_F_orig is not None:
             subject_features[f'F_{channel}_Original'] = features_F_orig

        features_Q_orig = calculate_quadratic_entropy(signal, m=QE_M, r=QE_R)
        if features_Q_orig is not None:
             subject_features[f'Q_{channel}_Original'] = features_Q_orig

        features_A_orig = calculate_quantile_graph_mjl(signal, nqg=QG_NQG, k=QG_K)
        if features_A_orig is not None:
             subject_features[f'A_{channel}_Original'] = features_A_orig

        features_I_orig = calculate_visibility_graph_iota(signal, fs)
        if features_I_orig is not None:
             subject_features[f'I_{channel}_Original'] = features_I_orig

        coeffs = decompose_signal(signal, wavelet=WAVELET_NAME, level=DECOMPOSITION_LEVEL)

        if coeffs is not None:
            relative_energies = calculate_relative_wavelet_energy(coeffs, BAND_LEVEL_MAP, FREQUENCY_BANDS)
            for band, energy in relative_energies.items():
                if energy is not None and np.isfinite(energy):
                     subject_features[f'E_{channel}_{band}'] = energy

            for band in FREQUENCY_BANDS:
                band_signal = reconstruct_band_signal(coeffs, band, BAND_LEVEL_MAP, wavelet=WAVELET_NAME, original_len=current_original_signal_len)

                if band_signal is not None and len(band_signal) == current_original_signal_len and not np.any(np.isnan(band_signal)) and not np.any(np.isinf(band_signal)):
                     features_F_band = katz_fd(band_signal)
                     if features_F_band is not None:
                          subject_features[f'F_{channel}_{band}'] = features_F_band

                     features_Q_band = calculate_quadratic_entropy(band_signal, m=QE_M, r=QE_R)
                     if features_Q_band is not None:
                          subject_features[f'Q_{channel}_{band}'] = features_Q_band

                     features_A_band = calculate_quantile_graph_mjl(band_signal, nqg=QG_NQG, k=QG_K)
                     if features_A_band is not None:
                          subject_features[f'A_{channel}_{band}'] = features_A_band

                     features_I_band = calculate_visibility_graph_iota(band_signal, fs)
                     if features_I_band is not None:
                          subject_features[f'I_{channel}_{band}'] = features_I_band

    valid_channel_names = list(valid_channels_data.keys())
    if len(valid_channel_names) >= 2:
        for i in range(len(valid_channel_names)):
            for j in range(i + 1, len(valid_channel_names)):
                ch1_name = valid_channel_names[i]
                ch2_name = valid_channel_names[j]
                sig1 = valid_channels_data[ch1_name]
                sig2 = valid_channels_data[ch2_name]

                sorted_channels = sorted([ch1_name, ch2_name])
                feature_name = f'C_{sorted_channels[0]}-{sorted_channels[1]}_Original'

                if feature_name not in subject_features:
                     wc_feature = calculate_wavelet_coherence(sig1, sig2, fs)
                     if wc_feature is not None:
                         subject_features[feature_name] = wc_feature

    return subject_features