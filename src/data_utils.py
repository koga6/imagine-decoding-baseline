from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np


LABEL_MAP: Dict[int, str] = {
    1: "apple",
    2: "bicycle",
    3: "brush",
    4: "cake",
    5: "clown",
    6: "cup",
    7: "desk",
    8: "foot",
    9: "mountain",
    10: "zebra",
}


def load_epochs(
    fif_path: str | Path,
    pick_types: str = "meg",
    resample_sfreq: Optional[float] = None,
):
    """
    Load epochs from FIF file and optionally pick channels / resample.
    """
    fif_path = Path(fif_path)
    if not fif_path.exists():
        raise FileNotFoundError(f"Epoch file not found: {fif_path}")

    epochs = mne.read_epochs(fif_path, preload=True, verbose=False)

    if pick_types == "meg":
        epochs = epochs.pick("meg")
    elif pick_types is not None:
        epochs = epochs.pick(pick_types)

    if resample_sfreq is not None:
        epochs = epochs.copy().resample(resample_sfreq)

    return epochs


def crop_epochs(
    epochs,
    tmin: float,
    tmax: float,
):
    """
    Return cropped copy of epochs.
    """
    return epochs.copy().crop(tmin=tmin, tmax=tmax)


def epochs_to_array(
    epochs,
    flatten: bool = True,
    clip_value: Optional[float] = None,
) -> np.ndarray:
    """
    Convert epochs to numpy array.
    Shape:
      - flatten=True  -> (n_trials, n_channels * n_times)
      - flatten=False -> (n_trials, n_channels, n_times)
    """
    X = epochs.get_data()  # (n_trials, n_channels, n_times)

    if clip_value is not None:
        X = np.clip(X, -clip_value, clip_value)

    if flatten:
        X = X.reshape(X.shape[0], -1)

    return X


def crop_and_vectorize(
    fif_path: str | Path,
    tmin: float,
    tmax: float,
    pick_types: str = "meg",
    resample_sfreq: Optional[float] = None,
    clip_value: Optional[float] = None,
):
    """
    Load epochs -> crop -> vectorize.
    """
    epochs = load_epochs(
        fif_path=fif_path,
        pick_types=pick_types,
        resample_sfreq=resample_sfreq,
    )
    epochs = crop_epochs(epochs, tmin=tmin, tmax=tmax)
    X = epochs_to_array(epochs, flatten=True, clip_value=clip_value)
    return epochs, X


def extract_labels_from_epochs(epochs) -> np.ndarray:
    """
    Extract class labels from epochs.events[:, 2].
    Assumes class ids are 1..10.
    """
    if epochs.events is None or epochs.events.shape[1] < 3:
        raise ValueError("Epochs does not contain valid events.")

    y = epochs.events[:, 2].astype(int)
    return y


def id_to_label(ids: np.ndarray | List[int]) -> List[str]:
    """
    Convert integer ids to string labels.
    """
    return [LABEL_MAP[int(i)] for i in ids]


def list_subject_dirs(root_dir: str | Path) -> List[Path]:
    """
    List subject directories sorted by name.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    return sorted([p for p in root_dir.iterdir() if p.is_dir()])


def get_subject_file_paths(subject_dir: str | Path) -> Tuple[Path, Path]:
    """
    Return localizer and imagine fif paths for one subject.
    """
    subject_dir = Path(subject_dir)
    subj = subject_dir.name

    localizer_path = subject_dir / f"{subj}_localizer-epo.fif"
    imagine_path = subject_dir / f"{subj}_imagine-epo.fif"

    return localizer_path, imagine_path