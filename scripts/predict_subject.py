from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data_utils import (  # noqa: E402
    crop_and_vectorize,
    extract_labels_from_epochs,
    get_subject_file_paths,
    id_to_label,
    list_subject_dirs,
)
from src.model_utils import build_classifier, fit_classifier  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "default_config.yaml"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Which split to run prediction on.",
    )
    return parser.parse_args()


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    split_dir = Path(cfg["paths"][f"{args.split}_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for subject_dir in list_subject_dirs(split_dir):
        subj = subject_dir.name
        localizer_path, imagine_path = get_subject_file_paths(subject_dir)

        print(f"Processing {subj} ...")

        localizer_epochs, X_train = crop_and_vectorize(
            fif_path=localizer_path,
            tmin=cfg["data"]["train_tmin"],
            tmax=cfg["data"]["train_tmax"],
            pick_types=cfg["data"]["pick_types"],
            resample_sfreq=cfg["data"].get("resample_sfreq", None),
            clip_value=cfg["data"].get("clip_value", None),
        )
        y_train = extract_labels_from_epochs(localizer_epochs)

        _, X_test = crop_and_vectorize(
            fif_path=imagine_path,
            tmin=cfg["data"]["test_tmin"],
            tmax=cfg["data"]["test_tmax"],
            pick_types=cfg["data"]["pick_types"],
            resample_sfreq=cfg["data"].get("resample_sfreq", None),
            clip_value=cfg["data"].get("clip_value", None),
        )

        model = build_classifier(cfg)
        model = fit_classifier(model, X_train, y_train)
        pred_ids = model.predict(X_test)
        pred_labels = id_to_label(pred_ids)

        for i, label in enumerate(pred_labels, start=1):
            rows.append({"ID": f"{subj}_{i}", "label": label})

    submission = pd.DataFrame(rows)
    submission_path = output_dir / cfg["predict"]["submission_name"]
    submission.to_csv(submission_path, index=False)

    print(f"Saved submission to: {submission_path}")
    print(submission.head())


if __name__ == "__main__":
    main()