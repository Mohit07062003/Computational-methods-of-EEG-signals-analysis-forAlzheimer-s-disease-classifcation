import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any

CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
            'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']

EXPECTED_SIGNAL_LENGTH = 8 * 128 # 1024 samples


def load_patient_data_subset(patient_path: str, channels: List[str], expected_length: int) -> Dict[str, np.ndarray | None]:
    eeg_data = {}
    for channel in channels:
        filepath = os.path.join(patient_path, f"{channel}.txt")
        if os.path.exists(filepath):
            try:
                data = np.loadtxt(filepath)
                if data.ndim > 1:
                    data = data.flatten()
                if len(data) > expected_length:
                    data = data[:expected_length]
                elif len(data) < expected_length:
                     eeg_data[channel] = None
                     continue

                eeg_data[channel] = data.astype(float)
            except Exception:
                 eeg_data[channel] = None
        else:
            eeg_data[channel] = None
    return eeg_data