import pywt
import numpy as np
from typing import List, Dict, Any

FS = 128

WAVELET_NAME = 'db4'
DECOMPOSITION_LEVEL = 4

BAND_LEVEL_MAP = {
    'delta':    [0],
    'theta':    [1],
    'alpha':    [2],
    'beta':     [3],
}
FREQUENCY_BANDS = list(BAND_LEVEL_MAP.keys())


def decompose_signal(signal: np.ndarray, wavelet: str = WAVELET_NAME, level: int = DECOMPOSITION_LEVEL) -> List[np.ndarray] | None:
    if signal is None or signal.ndim > 1 or len(signal) == 0:
        return None

    signal = signal.astype(float)

    try:
        min_len_for_level = 2**level
        if len(signal) <= min_len_for_level:
            return None

        coeffs = pywt.wavedec(signal, wavelet, level=level)
        return coeffs
    except ValueError:
        return None
    except Exception:
        return None


def reconstruct_band_signal(coeffs: List[np.ndarray] | None, band: str, band_level_map: Dict[str, List[int]], wavelet: str = WAVELET_NAME, original_len: int | None = None) -> np.ndarray | None:
    if coeffs is None or not isinstance(coeffs, list) or len(coeffs) == 0:
        return None

    if band not in band_level_map:
         return None

    target_coeffs_indices = band_level_map[band]

    reconstruction_coeffs = []
    all_coeffs_valid = True
    for i in range(len(coeffs)):
        if i in target_coeffs_indices:
            if i < len(coeffs) and coeffs[i] is not None:
                 reconstruction_coeffs.append(coeffs[i].copy())
            else:
                 all_coeffs_valid = False
                 break
        else:
            if i < len(coeffs) and coeffs[i] is not None:
                 reconstruction_coeffs.append(np.zeros_like(coeffs[i]))
            else:
                 all_coeffs_valid = False
                 break

    if not all_coeffs_valid or not reconstruction_coeffs:
         return None

    try:
        if original_len is not None:
             band_signal = pywt.waverec(reconstruction_coeffs, wavelet, an=original_len)
        else:
             band_signal = pywt.waverec(reconstruction_coeffs, wavelet)

        if original_len is not None and len(band_signal) != original_len:
             return None

        return band_signal.astype(float)
    except Exception:
        return None