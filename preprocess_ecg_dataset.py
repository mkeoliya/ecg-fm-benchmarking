#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
import subprocess
from pathlib import Path

# Add 'code' directory to python path
code_dir = Path(__file__).resolve().parent / "code"
sys.path.append(str(code_dir))

from clinical_ts.utils.ecg_utils import *
from clinical_ts.data.time_series_dataset_utils import *

DATASET_DIR = Path("/home/nvelingker/mkeoliya/kardia-lm/src/pipelines/ecg-fm-benchmarking/data")
TARGET_DIR = Path("processed")

CHANNEL_ITOS = "canonical"
channel_stoi_canonical = {
    "i": 0, "ii": 1, "iii": 2, "avr": 3, "avl": 4, "avf": 5, 
    "v1": 6, "v2": 7, "v3": 8, "v4": 9, "v5": 10, "v6": 11, 
    "vx": 12, "vy": 13, "vz": 14
}

def process_ptb_xl():
    print("Processing ptb-xl...")
    data_path = DATASET_DIR / "ptb-xl"
    target_folder = TARGET_DIR / "ptb-xl" / "records500"
    df, _, _, _ = prepare_data_ptb_xl(data_path=data_path, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=500, channel_itos=CHANNEL_ITOS, data_folder=target_folder)

def process_ptb():
    print("Processing ptb...")
    data_path = DATASET_DIR / "ptb"
    target_folder = TARGET_DIR / "ptb"
    df, _, _, _ = prepare_ptbv2(data_path=data_path, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=1000, channel_itos=CHANNEL_ITOS, data_folder=target_folder)

def process_ningbo():
    print("Processing ningbo...")
    data_path = DATASET_DIR / "ningbo"
    target_folder = TARGET_DIR / "ningbo"
    df, _, _, _ = prepare_data_ningbo(data_path=data_path, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=500, channel_itos=CHANNEL_ITOS, data_folder=target_folder)

def process_cpsc2018():
    print("Processing cpsc2018...")
    data_path = DATASET_DIR / "cpsc2018"
    target_folder = TARGET_DIR / "cpsc2018"
    df, _, _, _ = prepare_data_cpsc2018(data_path=data_path, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=500, channel_itos=CHANNEL_ITOS, data_folder=target_folder)

def process_cpsc_extra():
    print("Processing cpsc_extra...")
    data_path = DATASET_DIR / "cpsc_extra"
    target_folder = TARGET_DIR / "cpsc_extra"
    df, _, _, _ = prepare_data_cpsc_extra(data_path=data_path, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=500, channel_itos=CHANNEL_ITOS, data_folder=target_folder)

def process_georgia():
    print("Processing georgia...")
    data_path = DATASET_DIR / "georgia"
    target_folder = TARGET_DIR / "georgia"
    df, _, _, _ = prepare_data_georgia(data_path=data_path, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=500, channel_itos=CHANNEL_ITOS, data_folder=target_folder)

def process_sph():
    print("Processing sph...")
    data_path = DATASET_DIR / "sph"
    target_folder = TARGET_DIR / "sph"
    df, _, _, _ = prepare_data_sph(data_path=data_path, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=500, channel_itos=CHANNEL_ITOS, data_folder=target_folder)

def process_code15():
    print("Processing code15...")
    data_path = DATASET_DIR / "code15"
    target_folder = TARGET_DIR / "code15"
    df, _, _, _ = prepare_data_ribeiro_full(data_path=data_path, code15=True, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=400, channel_itos=CHANNEL_ITOS, data_folder=target_folder)

def process_chapman():
    print("Processing chapman...")
    data_path = DATASET_DIR / "chapman"
    target_folder = TARGET_DIR / "chapman"
    df, _, _, _ = prepare_data_chapman(data_path=data_path, denoised=False, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=500, channel_itos=CHANNEL_ITOS, data_folder=target_folder)

def process_mimic():
    print("Processing mimic...")
    data_path = DATASET_DIR / "mimic"
    target_folder = TARGET_DIR / "mimic"
    df, _, _, _ = prepare_mimicecg(data_path=data_path, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=500, channel_itos=CHANNEL_ITOS, data_folder=target_folder)
    
    # Run the mimic_preprocessing script
    print("Running mimic_preprocessing.py...")
    subprocess.run([sys.executable, "mimic_preprocessing.py"], check=True, cwd=Path(__file__).resolve().parent)

def process_echonext():
    print("Processing echonext...")
    data_path = DATASET_DIR / "echonext"
    target_folder = TARGET_DIR / "echonext"
    df, _, _, _ = prepare_data_echonext(data_path=data_path, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=250, channel_itos=CHANNEL_ITOS, data_folder=target_folder)

def process_zzu_pecg():
    print("Processing zzu_pecg...")
    data_path = DATASET_DIR / "zzu_pecg"
    target_folder = TARGET_DIR / "zzu_pecg"
    df, _, _, _ = prepare_data_zzu_pecg(data_path=data_path, target_folder=target_folder)
    reformat_as_memmap(df=df, target_filename=target_folder/"memmap.npy", fs=500, channel_itos=CHANNEL_ITOS, data_folder=target_folder)

def process_heedb():
    print("Processing heedb...")
    data_path = DATASET_DIR / "heedb"
    target_folder = TARGET_DIR / "heedb"
    partitions = [
        'S0001-1987', 'S0001-1988', 'S0001-1994', 'S0001-2006', 'S0001-2007', 
        'S0001-2011', 'S0001-2013', 'S0001-2015', 'S0001-2016', 'S0001-2017', 
        'S0001-2018', 'S0001-2019', 'S0001-2020', 'S0001-2021', 'S0002-1993', 
        'S0002-1997', 'S0002-1998', 'S0002-2000', 'S0002-2003', 'S0002-2004', 
        'S0002-2008', 'S0002-2009', 'S0002-2013', 'S0002-2015', 'S0002-2016', 
        'S0002-2018', 'S0002-2020', 'S0002-2022', 'S0003-1990', 'S0003-1994', 
        'S0003-2009', 'S0003-2010', 'S0003-2013', 'S0004-2019'
    ]
    for partition in partitions:
        print(f"  Partition: {partition}")
        df, _, _, _ = prepare_heedb(data_path=data_path, target_folder=target_folder, partition=partition)
        part_target_folder = target_folder / partition
        reformat_as_memmap(df=df, target_filename=part_target_folder/"memmap.npy", fs=240, channel_itos=CHANNEL_ITOS, data_folder=part_target_folder)

DATASET_PROCESSORS = {
    "ptb-xl": process_ptb_xl,
    "ptb": process_ptb,
    "ningbo": process_ningbo,
    "cpsc2018": process_cpsc2018,
    "cpsc_extra": process_cpsc_extra,
    "georgia": process_georgia,
    "sph": process_sph,
    "code15": process_code15,
    "chapman": process_chapman,
    "mimic": process_mimic,
    "echonext": process_echonext,
    "zzu_pecg": process_zzu_pecg,
    "heedb": process_heedb,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process ECG datasets")
    parser.add_argument(
        "datasets", 
        nargs="+", 
        choices=list(DATASET_PROCESSORS.keys()) + ["all"],
        help="Which dataset(s) to process, or 'all'"
    )
    args = parser.parse_args()

    datasets_to_process = args.datasets
    if "all" in datasets_to_process:
        datasets_to_process = list(DATASET_PROCESSORS.keys())
    
    # Remove duplicates but preserve order
    datasets_to_process = list(dict.fromkeys(datasets_to_process))

    print(f"Datasets scheduled for processing: {', '.join(datasets_to_process)}")
    print("-" * 50)
    for ds in datasets_to_process:
        DATASET_PROCESSORS[ds]()
        print("Done processsing", ds)
        print("-" * 50)
    print("All tasks completed.")
