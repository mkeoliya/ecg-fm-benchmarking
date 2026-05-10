#!/usr/bin/env python
"""Export ECG ID and label pairs used by linear probing."""

from __future__ import annotations

import argparse
import csv
import pathlib
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


UNSUPPORTED_DATASETS = {}

DATASET_ALIASES = {
    "code15": "code15_diag",
}

DATASET_ROOTS = {
    "ptbxl_super": Path("ptb-xl") / "records500",
    "ptbxl_sub": Path("ptb-xl") / "records500",
    "ptbxl_all": Path("ptb-xl") / "records500",
    "code15_diag": Path("code15"),
}

MIMIC_REGRESSION_TARGETS = 35

DATASET_SPECS = {
    "ptb": {"label_col": "label"},
    "ningbo": {
        "label_names_key": "label_filtered",
        "label_col": "label_filtered_numeric",
    },
    "cpsc2018": {
        "label_names_key": "label_filtered",
        "label_col": "label_filtered_numeric",
        "min_data_length": 5000,
    },
    "cpsc_extra": {
        "label_names_key": "label_filtered",
        "label_col": "label_filtered_numeric",
        "min_data_length": 5000,
    },
    "georgia": {
        "label_names_key": "label_filtered",
        "label_col": "label_filtered_numeric",
        "min_data_length": 5000,
    },
    "chapman": {
        "label_names_key": "label_all_filtered",
        "label_col": "label_all_filtered_numeric",
    },
    "sph": {
        "label_names_key": "label_primary_filtered",
        "label_col": "label_primary_filtered_numeric",
    },
    "code15_diag": {
        "label_col": "label",
        "min_data_length": 4000,
        "require_nonnegative_strat_fold": True,
    },
    "ptbxl_super": {
        "label_names_key": "label_diag_superclass",
        "label_col": "label_diag_superclass_filtered_numeric",
    },
    "ptbxl_sub": {
        "label_names_key": "label_diag_subclass",
        "label_col": "label_diag_subclass_filtered_numeric",
    },
    "ptbxl_all": {
        "label_names_key": "label_all",
        "label_col": "label_all_filtered_numeric",
    },
    "echonext": {
        "label_col": "label",
        "split_col": "split",
        "valid_splits": {"train", "val", "test"},
    },
    "zzu_pecg": {
        "label_names_key": "aha_description_filtered",
        "label_col": "aha_description_filtered_numeric",
        "min_data_length": 5000,
    },
}

SPH_PRIMARY_LABEL_NAMES = {
    "1": "Normal ECG",
    "21": "Sinus tachycardia",
    "22": "Sinus bradycardia",
    "23": "Sinus arrhythmia",
    "30": "Atrial premature complexes",
    "31": "Atrial premature complexes nonconducted",
    "36": "Junctional premature complexes",
    "37": "Junctional escape complexes",
    "50": "Atrial fibrillation",
    "51": "Atrial flutter",
    "54": "Junctional tachycardia",
    "60": "Ventricular premature complexes",
    "80": "Short PR interval",
    "81": "AV conduction ratio N:D",
    "82": "Prolonged PR interval",
    "83": "Second-degree AV block Mobitz type I",
    "84": "Second-degree AV block Mobitz type II",
    "85": "2:1 AV block",
    "86": "AV block varying conduction",
    "87": "Advanced AV block",
    "88": "Complete AV block",
    "101": "Left anterior fascicular block",
    "102": "Left posterior fascicular block",
    "104": "Left bundle-branch block",
    "105": "Incomplete right bundle-branch block",
    "106": "Right bundle-branch block",
    "108": "Ventricular preexcitation",
    "120": "Right-axis deviation",
    "121": "Left-axis deviation",
    "125": "Low voltage",
    "140": "Left atrial enlargement",
    "142": "Left ventricular hypertrophy",
    "143": "Right ventricular hypertrophy",
    "145": "ST deviation",
    "146": "ST deviation with T-wave change",
    "147": "T-wave abnormality",
    "148": "Prolonged QT interval",
    "152": "TU fusion",
    "153": "ST-T change due to ventricular hypertrophy",
    "155": "Early repolarization",
    "160": "Anterior MI",
    "161": "Inferior MI",
    "165": "Anteroseptal MI",
    "166": "Extensive anterior MI",
}

LABEL_NAME_OVERRIDES = {
    "sph": SPH_PRIMARY_LABEL_NAMES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ECG ID and label pairs from a processed probing dataset."
    )
    parser.add_argument(
        "--dataset",
        default="ptb",
        help="Processed dataset name.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Processed dataset folder, e.g. processed/ptb. Defaults to --data-dir/--dataset.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("processed"),
        help="Root folder containing processed datasets.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Deprecated. Split CSVs are written to this path's parent directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("linear/probing_pairs"),
        help="Output directory used when --output is omitted.",
    )
    parser.add_argument(
        "--pairs-only",
        action="store_true",
        help="Write only ecg_id, label, and split columns.",
    )
    return parser.parse_args()


def normalize_dataset(dataset: str) -> str:
    dataset = dataset.lower()
    return DATASET_ALIASES.get(dataset, dataset)


def default_dataset_root(data_dir: Path, dataset: str) -> Path:
    return data_dir / DATASET_ROOTS.get(dataset, Path(dataset))


def load_label_mapping(dataset_root: Path) -> object:
    lbl_itos_pkl = dataset_root / "lbl_itos.pkl"
    lbl_itos_npy = dataset_root / "lbl_itos.npy"
    if lbl_itos_pkl.exists():
        with lbl_itos_pkl.open("rb") as infile:
            return pickle.load(infile)
    if lbl_itos_npy.exists():
        return np.load(lbl_itos_npy, allow_pickle=True)
    raise FileNotFoundError(f"Label mapping file not found in {dataset_root}")


def load_processed_dataset(dataset_root: Path) -> tuple[pd.DataFrame, object]:
    # Some metadata pickles were written by a pathlib variant that stores paths
    # under pathlib._local. Alias it for Python versions where that is not a
    # real importable module.
    sys.modules.setdefault("pathlib._local", pathlib)

    df_path = dataset_root / "df_memmap.pkl"
    if not df_path.exists():
        df_path = dataset_root / "df.pkl"
    if not df_path.exists():
        raise FileNotFoundError(f"Dataset metadata file not found in {dataset_root}")

    df = pd.read_pickle(df_path)
    lbl_itos = load_label_mapping(dataset_root)
    return df, lbl_itos


def label_indices(value: object) -> list[int]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(idx) for idx in value]
    if pd.isna(value):
        return []
    return [int(value)]


def multihot_encode(label_indices: Iterable[int], num_classes: int) -> list[int]:
    label = np.zeros(num_classes, dtype=np.int64)
    for idx in label_indices:
        idx = int(idx)
        if idx < 0 or idx >= num_classes:
            raise ValueError(f"Label index {idx} is outside [0, {num_classes})")
        label[idx] = 1
    return label.tolist()


def path_stem(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return Path(text).stem if text else text


def derive_ecg_ids(df: pd.DataFrame) -> pd.Series:
    for col in ("ecg_id", "data_original", "data", "filename"):
        if col in df.columns:
            values = df[col]
            if col == "data" and pd.api.types.is_numeric_dtype(values):
                continue
            ids = values.apply(path_stem)
            if ids.astype(bool).all():
                return ids
    raise ValueError("Could not derive ecg_id from columns: ecg_id, data_original, data, filename")


def add_linear_probe_split(df: pd.DataFrame, spec: dict) -> pd.Series:
    split_col = spec.get("split_col")
    if split_col is not None and split_col in df.columns:
        return df[split_col].astype(str)

    if "strat_fold" not in df.columns:
        return pd.Series(["all"] * len(df), index=df.index)

    max_fold_id = df["strat_fold"].max()
    split = pd.Series(["train"] * len(df), index=df.index)
    split.loc[df["strat_fold"] == max_fold_id - 1] = "val"
    split.loc[df["strat_fold"] == max_fold_id] = "test"
    return split


def select_label_names(lbl_itos: object, spec: dict, dataset: str) -> list[str]:
    key = spec.get("label_names_key")
    if key is None:
        if isinstance(lbl_itos, dict):
            raise ValueError(f"{dataset} requires label_names_key because lbl_itos is a dict")
        values = lbl_itos
    else:
        if not isinstance(lbl_itos, dict) or key not in lbl_itos:
            raise ValueError(f"{dataset} label mapping does not contain key {key!r}")
        values = lbl_itos[key]
    label_names = [str(label) for label in np.array(values)]
    overrides = LABEL_NAME_OVERRIDES.get(dataset)
    if overrides is None:
        return label_names
    return [overrides.get(label, label) for label in label_names]



def prepare_dataset_for_pairs(
    df: pd.DataFrame,
    spec: dict,
    dataset: str,
) -> pd.DataFrame:
    df = df.copy()

    if spec.get("valid_splits") is not None:
        split_col = spec.get("split_col", "split")
        if split_col not in df.columns:
            raise ValueError(f"{dataset} metadata does not contain split column {split_col!r}")
        df = df[df[split_col].isin(spec["valid_splits"])].copy()

    if spec.get("require_nonnegative_strat_fold"):
        if "strat_fold" not in df.columns:
            raise ValueError(f"{dataset} metadata does not contain strat_fold")
        df = df[df["strat_fold"] >= 0].copy()

    min_data_length = spec.get("min_data_length")
    if min_data_length is not None:
        if "data_length" not in df.columns:
            raise ValueError(f"{dataset} metadata does not contain data_length")
        df = df[df["data_length"] >= min_data_length].copy()

    return df


def build_pairs(dataset: str, df: pd.DataFrame, lbl_itos: object) -> pd.DataFrame:
    spec = DATASET_SPECS[dataset]
    df = prepare_dataset_for_pairs(df, spec, dataset)
    label_col = spec["label_col"]
    if label_col not in df.columns:
        raise ValueError(f"{dataset} metadata does not contain label column {label_col!r}")

    label_names = select_label_names(lbl_itos, spec, dataset)
    indices = df[label_col].apply(label_indices)
    label_matrix = np.array(
        [multihot_encode(row_indices, len(label_names)) for row_indices in indices],
        dtype=np.int64,
    )

    pairs = pd.DataFrame()
    pairs["ecg_id"] = derive_ecg_ids(df)
    pairs["split"] = add_linear_probe_split(df, spec)
    pairs["label"] = indices.apply(lambda xs: ";".join(label_names[i] for i in xs))
    for class_idx, class_name in enumerate(label_names):
        pairs[class_name] = label_matrix[:, class_idx]
    return pairs


def write_outputs(
    pairs: pd.DataFrame,
    dataset: str,
    output_dir: Path,
    pairs_only: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    out = pairs.copy()
    if pairs_only:
        out = out[["ecg_id", "split", "label"]]

    paths = []
    for split in ("train", "val", "test"):
        split_out = out[out["split"] == split].drop(columns=["split"])
        split_path = output_dir / f"{dataset}_{split}.csv"
        split_out.to_csv(split_path, index=False)
        paths.append(split_path)
    return paths


def load_mimic_benchmark(dataset_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    sys.modules.setdefault("pathlib._local", pathlib)

    benchmark_path = dataset_root / "df_mimic_benchmark.pkl"
    if not benchmark_path.exists():
        raise FileNotFoundError(f"MIMIC benchmark labels not found: {benchmark_path}")

    lbl_itos_path = dataset_root / "lbl_itos_mimic.npy"
    if not lbl_itos_path.exists():
        raise FileNotFoundError(f"MIMIC label names not found: {lbl_itos_path}")

    return pd.read_pickle(benchmark_path), np.load(lbl_itos_path, allow_pickle=True)


def value_for_csv(value: object) -> object:
    if pd.isna(value):
        return ""
    return value


def classification_value_for_csv(value: object) -> object:
    if pd.isna(value):
        return ""
    return int(value)


def mimic_ecg_id(row: pd.Series) -> str:
    if "file_name" in row and not pd.isna(row["file_name"]):
        return Path(str(row["file_name"])).stem
    if "study_id" in row and not pd.isna(row["study_id"]):
        return str(int(row["study_id"]))
    if "data" in row and not pd.isna(row["data"]):
        return str(row["data"])
    raise ValueError("Could not derive MIMIC ecg_id from file_name, study_id, or data")


def write_mimic_target_files(
    df: pd.DataFrame,
    label_names: np.ndarray,
    output_dir: Path,
) -> tuple[list[Path], pd.Series]:
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = [str(label) for label in label_names[:-MIMIC_REGRESSION_TARGETS]]
    regression_names = [str(label) for label in label_names[-MIMIC_REGRESSION_TARGETS:]]

    paths = []
    split_series = add_linear_probe_split(df, {})

    for split in ("train", "val", "test"):
        split_df = df[split_series == split]

        classification_path = output_dir / f"mimic_classification_{split}.csv"
        with classification_path.open("w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["ecg_id", "label", *class_names])
            for _, row in split_df.iterrows():
                label_all = row["label_all"]
                cls_values = label_all[:-MIMIC_REGRESSION_TARGETS]
                positive_labels = [
                    name
                    for name, value in zip(class_names, cls_values)
                    if not pd.isna(value) and float(value) == 1.0
                ]
                writer.writerow(
                    [
                        mimic_ecg_id(row),
                        ";".join(positive_labels),
                        *[classification_value_for_csv(value) for value in cls_values],
                    ]
                )
        paths.append(classification_path)

        regression_path = output_dir / f"mimic_regression_{split}.csv"
        with regression_path.open("w", newline="") as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["ecg_id", *regression_names])
            for _, row in split_df.iterrows():
                label_all = row["label_all"]
                reg_values = label_all[-MIMIC_REGRESSION_TARGETS:]
                writer.writerow([mimic_ecg_id(row), *[value_for_csv(value) for value in reg_values]])
        paths.append(regression_path)

    return paths, split_series


def main() -> None:
    args = parse_args()
    requested_dataset = args.dataset.lower()
    dataset = normalize_dataset(requested_dataset)
    if dataset in UNSUPPORTED_DATASETS:
        raise NotImplementedError(UNSUPPORTED_DATASETS[dataset])
    if dataset != "mimic" and dataset not in DATASET_SPECS:
        supported = ", ".join(sorted(DATASET_SPECS))
        raise ValueError(f"Unknown dataset {requested_dataset!r}. Supported datasets: {supported}")

    dataset_root = args.data if args.data is not None else default_dataset_root(args.data_dir, dataset)
    output_dir = args.output.parent if args.output is not None else args.output_dir

    if dataset == "mimic":
        df, lbl_itos = load_mimic_benchmark(dataset_root)
        split_paths, split_series = write_mimic_target_files(df, lbl_itos, output_dir)
        split_counts = split_series.value_counts().reindex(["train", "val", "test"], fill_value=0)
        print(f"Dataset: {dataset}")
        print(f"Rows: {len(df)}")
        print(
            "Splits: "
            + ", ".join(f"{split}={int(count)}" for split, count in split_counts.items())
        )
        print("Files:")
        for path in split_paths:
            print(f"  {path}")
        return

    df, lbl_itos = load_processed_dataset(dataset_root)
    pairs = build_pairs(dataset, df, lbl_itos)
    split_paths = write_outputs(pairs, requested_dataset, output_dir, args.pairs_only)

    split_counts = pairs["split"].value_counts().reindex(["train", "val", "test"], fill_value=0)
    print(f"Dataset: {dataset}")
    print(f"Rows: {len(pairs)}")
    print(
        "Splits: "
        + ", ".join(f"{split}={int(count)}" for split, count in split_counts.items())
    )
    print("Files:")
    for path in split_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
