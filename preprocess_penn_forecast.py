#!/usr/bin/env python
"""Preprocess binary WFDB CSV splits into this repo's memmap dataset format.

The default arguments match the Penn forecast layout, but the script itself only
assumes that each CSV has a record-name column and a binary label column.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from tqdm.auto import tqdm


code_dir = Path(__file__).resolve().parent / "code"
sys.path.append(str(code_dir))

from clinical_ts.data.time_series_dataset_utils import (  # noqa: E402
    dataset_get_stats,
    reformat_as_memmap,
    save_dataset,
)
from clinical_ts.utils.signal_utils import channel_stoi_canonical, resample_data  # noqa: E402


DEFAULT_SPLIT_DIR = Path("/srv/shared_home/common-data/arpa-h/ca/melp_split/penn_forecast")
DEFAULT_MELP_SPLIT_ROOT = Path("/srv/shared_home/common-data/arpa-h/ca/melp_split")
DEFAULT_RECORD_ROOT = Path("/srv/shared_home/common-data/arpa-h/ca/penn/wfdb")
DEFAULT_OUT_DIR = Path("processed/penn_forecast")
CHANNEL_ITOS = "canonical"
DEFAULT_SPLIT_TO_FOLD = {
    "train": 0,
    "val": 8,
    "valid": 8,
    "validation": 8,
    "test": 9,
}


def parse_label(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    value_norm = str(value).strip().lower()
    if value_norm in {"true", "1", "yes", "y", "positive", "pos"}:
        return 1
    if value_norm in {"false", "0", "no", "n", "negative", "neg"}:
        return 0
    raise ValueError(f"Cannot parse label value {value!r} as binary.")


def parse_split_folds(items):
    split_to_fold = dict(DEFAULT_SPLIT_TO_FOLD)
    for item in items or []:
        if ":" not in item:
            raise ValueError(f"Expected --split-fold as SPLIT:FOLD, got {item!r}")
        split, fold = item.split(":", 1)
        split_to_fold[split] = int(fold)
    return split_to_fold


def infer_split(csv_path):
    stem = Path(csv_path).stem.lower()
    if "train" in stem:
        return "train"
    if "valid" in stem:
        return "valid"
    if "val" in stem:
        return "val"
    if "test" in stem:
        return "test"
    return stem


def csv_specs(args):
    if args.csv:
        specs = []
        for item in args.csv:
            if "=" in item:
                split, csv_path = item.split("=", 1)
            else:
                csv_path = item
                split = infer_split(csv_path)
            specs.append((split, Path(csv_path)))
        return specs

    split_dir = args.split_dir
    if args.melp_split_name is not None:
        split_dir = args.melp_split_root / args.melp_split_name

    csv_template = args.csv_template
    if csv_template is None:
        csv_template = f"{split_dir.name}_{{split}}.csv"

    return [
        (split, split_dir / csv_template.format(split=split))
        for split in args.splits
    ]


def load_split(csv_path, split, split_to_fold, args):
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing split CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {args.name_col, args.label_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    if args.limit is not None:
        df = df.sample(n=min(args.limit, len(df)), random_state=args.seed)

    df = df.copy()
    df["split"] = split
    df["strat_fold"] = split_to_fold.get(split, 0)
    return df


def strip_wfdb_suffix(path):
    path = Path(path)
    if path.suffix.lower() in {".hea", ".dat"}:
        return path.with_suffix("")
    return path


def resolve_record_path(record_root, record_name):
    record_path = strip_wfdb_suffix(Path(str(record_name)))
    if record_path.is_absolute():
        return record_path
    return strip_wfdb_suffix(record_root / record_path)


def header_path(record_path):
    return Path(str(record_path) + ".hea")


def safe_output_stem(record_name):
    path = strip_wfdb_suffix(Path(str(record_name)))
    parts = path.parts[1:] if path.is_absolute() else path.parts
    return "__".join(parts)


def read_record(record_path, target_fs, channels, clip_amp):
    sigbufs, header = wfdb.rdsamp(str(record_path))
    sigbufs = sigbufs.astype(np.float32, copy=False)

    if np.any(np.isnan(sigbufs)):
        sigbufs = np.nan_to_num(sigbufs, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_amp is not None:
        sigbufs = np.clip(sigbufs, -clip_amp, clip_amp)

    return resample_data(
        sigbufs=sigbufs,
        channel_labels=header["sig_name"],
        fs=header["fs"],
        target_fs=target_fs,
        channels=channels,
        channel_stoi=channel_stoi_canonical,
    )


def process_record_task(task):
    name = task["name"]
    split = task["split"]
    record_path = resolve_record_path(task["record_root"], name)

    if not header_path(record_path).exists():
        raise FileNotFoundError(f"Missing WFDB header for {name}: {header_path(record_path)}")

    rel_npy = Path("npys") / split / f"{safe_output_stem(name)}.npy"
    out_npy = task["out_dir"] / rel_npy
    out_npy.parent.mkdir(parents=True, exist_ok=True)

    if out_npy.exists() and not task["overwrite"]:
        data = np.load(out_npy, allow_pickle=True)
    else:
        data = read_record(record_path, task["target_fs"], task["channels"], task["clip_amp"])
        data = data.astype(task["dtype"], copy=False)
        np.save(out_npy, data)

    return {
        "idx": task["idx"],
        "row": {
            "record_name": name,
            "record_path": record_path,
            "data": rel_npy,
            "data_length": int(data.shape[0]),
            "data_mean": np.mean(data, axis=0),
            "data_std": np.std(data, axis=0),
            "label": task["label"],
            "split": split,
            "strat_fold": task["strat_fold"],
            "dataset": task["dataset_name"],
        },
    }


def run_record_task(task, skip_errors):
    try:
        return process_record_task(task), None
    except Exception as exc:
        if skip_errors:
            return None, f"Failed to process {task['name']}: {exc}"
        raise


def preprocess(args):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "npys").mkdir(parents=True, exist_ok=True)

    split_to_fold = parse_split_folds(args.split_fold)
    split_frames = [
        load_split(csv_path, split, split_to_fold, args)
        for split, csv_path in csv_specs(args)
    ]
    source_df = pd.concat(split_frames, ignore_index=True)
    dataset_name = args.dataset_name or args.out_dir.name

    tasks = []
    for idx, row in source_df.iterrows():
        tasks.append(
            {
                "idx": idx,
                "name": str(row[args.name_col]),
                "label": parse_label(row[args.label_col]),
                "split": str(row["split"]),
                "strat_fold": int(row["strat_fold"]),
                "dataset_name": dataset_name,
                "record_root": args.record_root,
                "out_dir": args.out_dir,
                "target_fs": args.target_fs,
                "channels": args.channels,
                "clip_amp": args.clip_amp,
                "dtype": args.dtype,
                "overwrite": args.overwrite,
            }
        )

    results = []
    failures = []
    if args.workers <= 1:
        iterator = (run_record_task(task, args.skip_errors) for task in tasks)
        for result, failure in tqdm(iterator, total=len(tasks), desc=dataset_name):
            if failure is not None:
                failures.append(failure)
            elif result is not None:
                results.append(result)
    else:
        executor_cls = ThreadPoolExecutor if args.worker_backend == "threads" else ProcessPoolExecutor
        with executor_cls(max_workers=args.workers) as executor:
            futures = [executor.submit(run_record_task, task, args.skip_errors) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc=dataset_name):
                result, failure = future.result()
                if failure is not None:
                    failures.append(failure)
                elif result is not None:
                    results.append(result)

    if failures:
        failure_path = args.out_dir / "failures.txt"
        failure_path.write_text("\n".join(failures) + "\n")
        print(f"Skipped {len(failures)} records. See {failure_path}.")

    results = sorted(results, key=lambda item: item["idx"])
    df = pd.DataFrame([item["row"] for item in results])
    if df.empty:
        raise RuntimeError("No records were successfully preprocessed.")

    lbl_itos = np.array([args.positive_label_name])
    mean, std = dataset_get_stats(df)
    save_dataset(df, lbl_itos=lbl_itos, mean=mean, std=std, target_root=args.out_dir)
    reformat_as_memmap(
        df=df,
        target_filename=args.out_dir / "memmap.npy",
        fs=args.target_fs,
        channel_itos=CHANNEL_ITOS,
        data_folder=args.out_dir,
        max_len=args.max_len,
        delete_npys=not args.keep_npy,
        batch_length=args.batch_length,
    )

    counts = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(f"Wrote {len(df)} records to {args.out_dir}")
    print("Label counts by split:")
    print(counts)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Preprocess binary WFDB CSV splits into memmap.npy/df_memmap.pkl."
    )
    parser.add_argument(
        "--csv",
        action="append",
        help="CSV path or SPLIT=CSV path. Repeat for multiple splits. Overrides --split-dir/--csv-template.",
    )
    parser.add_argument("--melp-split-name", default=None, help="Name under --melp-split-root; files default to NAME_{split}.csv.")
    parser.add_argument("--melp-split-root", type=Path, default=DEFAULT_MELP_SPLIT_ROOT)
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--csv-template", default=None, help="Defaults to '<split-dir-name>_{split}.csv'.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--split-fold", action="append", help="Override fold assignment as SPLIT:FOLD.")
    parser.add_argument("--record-root", "--wfdb-dir", dest="record_root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--name-col", default="NAME")
    parser.add_argument("--label-col", default="Label")
    parser.add_argument("--positive-label-name", default="abnormal")
    parser.add_argument("--target-fs", type=int, default=250)
    parser.add_argument("--channels", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for WFDB read/resample/npy creation.")
    parser.add_argument("--worker-backend", choices=["threads", "processes"], default="threads")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--clip-amp", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=900_000_000)
    parser.add_argument("--batch-length", type=int, default=0, help="Batch size in timesteps for final memmap packing; 0 disables batching.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum rows to process per CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used when --limit samples rows.")
    parser.add_argument("--keep-npy", action="store_true", help="Keep intermediate per-record .npy files.")
    parser.add_argument("--overwrite", action="store_true", help="Recreate existing intermediate .npy files.")
    parser.add_argument("--skip-errors", action="store_true", help="Skip missing/corrupt records and continue.")
    return parser


if __name__ == "__main__":
    preprocess(build_parser().parse_args())
