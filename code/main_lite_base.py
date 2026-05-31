"""
Base classes and utilities for main_lite: Main_Lite, FMWrapperBase, and multihot_encode.
"""
import torch
from torch import nn
import lightning.pytorch as lp
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import dataclasses
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

from clinical_ts.utils.eval_utils_cafa import multiclass_roc_curve
from clinical_ts.utils.eval_utils_regression import regression_metrics
from clinical_ts.data.time_series_dataset import *
from clinical_ts.data.time_series_dataset_utils import *
from clinical_ts.data.time_series_dataset_transforms import *
from clinical_ts.utils.bootstrap_utils import empirical_bootstrap
from clinical_ts.utils.schedulers import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_invsqrt_decay_schedule_with_warmup,
)

from clinical_ts.data.time_series_dataset import TimeSeriesDataset
from clinical_ts.data.time_series_dataset_utils import load_wfdb_dataset
# Utility function

def multihot_encode(x, num_classes):
    res = np.zeros(num_classes,dtype=np.float32)
    for y in x:
        res[y]=1
    return res

# at this scope to avoid pickle issues
def mcrc_flat(targs,preds,classes):
    _,_,res = multiclass_roc_curve(targs,preds,classes=classes)
    return np.array(list(res.values()))

def regression_flat(targs, preds, metrics=["mae"], target_names=None):
    res = regression_metrics(targs, preds, metrics=metrics, target_names=target_names)

    n_targets = targs.shape[1]
    if target_names is None:
        target_names = [str(i) for i in range(n_targets)]
    
    ordered_values = []

    for metric in metrics:
        ordered_values.append(res[metric])
    
    for target_name in target_names:
        for metric in metrics:
            ordered_values.append(res[f"{target_name}_{metric}"])

    return np.array(ordered_values)

def _is_empty_arg(value):
    return value is None or str(value).strip() == ""

def _resolve_column(columns, explicit, candidates):
    columns = list(columns)
    if not _is_empty_arg(explicit):
        if explicit not in columns:
            raise ValueError(f"Column {explicit!r} not found. Available columns: {columns}")
        return explicit

    by_lower = {str(c).lower(): c for c in columns}
    for candidate in candidates:
        resolved = by_lower.get(candidate.lower())
        if resolved is not None:
            return resolved
    return None

def _split_values(split_name):
    if split_name == "val":
        return {"val", "valid", "validation"}
    return {split_name}

def _split_requested(split_name, requested):
    if _is_empty_arg(requested):
        requested = "test"
    requested = {item.strip().lower() for item in str(requested).split(",") if item.strip()}
    return "all" in requested or bool(_split_values(split_name).intersection(requested))

def _csv_paths(value):
    return [Path(item.strip()) for item in str(value).split(",") if item.strip()]

####################################################################################################
# Embedding Dataset for precomputed embeddings
####################################################################################################
class EmbeddingDataset(TimeSeriesDataset):
    """Dataset that returns precomputed embeddings and labels"""
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx],
            "label": self.labels[idx]
        }
    
    def get_id_mapping(self):
        """Return identity mapping for aggregation"""
        return np.arange(len(self))

    def aggregate_predictions(self, preds,targs=None,idmap=None,aggregate_fn = np.mean,verbose=False):
        '''
        aggregates potentially multiple predictions per sample (can also pass targs for convenience)
        idmap: idmap as returned by TimeSeriesCropsDataset's get_id_mapping (uses self.get_id_mapping by default)
        preds: ordered predictions as returned by learn.get_preds()
        aggregate_fn: function that is used to aggregate multiple predictions per sample (most commonly np.amax or np.mean)
        '''
        idmap = self.get_id_mapping() if idmap is None else idmap
        if(idmap is not None and len(idmap)!=len(np.unique(idmap))):
            if(verbose):
                print("aggregating predictions...")
            preds_aggregated = []
            targs_aggregated = []
            for i in np.unique(idmap):
                preds_local = preds[np.where(idmap==i)[0]]
                preds_aggregated.append(aggregate_fn(preds_local,axis=0))
                if targs is not None:
                    targs_local = targs[np.where(idmap==i)[0]]
                    #assert(np.all(targs_local==targs_local[0])) #all labels have to agree
                    assert(np.all([np.array_equal(t, targs_local[0], equal_nan=True) for t in targs_local])) #all labels have to agree (including nans)
                    targs_aggregated.append(targs_local[0])
            if(targs is None):
                return np.array(preds_aggregated)
            else:
                return np.array(preds_aggregated),np.array(targs_aggregated)
        else:
            if(targs is None):
                return preds
            else:
                return preds,targs

# FMWrapperBase
class FMWrapperBase(nn.Module):
    def __init__(self, num_classes, num_output_tokens):
        super().__init__()
        self.num_classes = num_classes
        self.num_output_tokens = num_output_tokens

    def get_model_transforms(self, tfms_lst):
        return tfms_lst

# Main_Lite base class
class Main_Lite(lp.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.lr = self.hparams.lr
        print(hparams)
    # ... rest of Main_Lite definition ... 

    def forward(self, x, **kwargs):
        x[torch.isnan(x)]=0
        x = self.model(x, **kwargs)
        x = torch.nan_to_num(x, nan=0.0)
        return x

    def on_validation_epoch_end(self):
        for i in range(len(self.val_preds)):
            self.on_valtest_epoch_eval({"preds":self.val_preds[i], "targs":self.val_targs[i]}, dataloader_idx=i, test=False)
            self.val_preds[i].clear()
            self.val_targs[i].clear()

    def on_test_epoch_end(self):
        for i in range(len(self.test_preds)):
            self.on_valtest_epoch_eval({"preds":self.test_preds[i], "targs":self.test_targs[i]}, dataloader_idx=i, test=True)
            self.test_preds[i].clear()
            self.test_targs[i].clear()

    def _bootstrap_results_path(self):
        log_dir = getattr(self.trainer, "log_dir", None)
        if log_dir is None:
            trainer_logger = getattr(self.trainer, "logger", None)
            log_dir = getattr(trainer_logger, "log_dir", None)
        if log_dir is None:
            log_dir = getattr(self.hparams, "output_path", ".")
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / "bootstrap_iteration_results.csv"

    def write_bootstrap_results(self, bootstrap_results, split, aggregation, dataloader_idx, auc_suffix):
        if not bootstrap_results:
            return

        output_path = self._bootstrap_results_path()
        write_header = not output_path.exists()
        fieldnames = [
            "epoch",
            "split",
            "aggregation",
            "dataloader_idx",
            "bootstrap_iteration",
            "metric",
            "logged_metric",
            "value",
        ]

        with output_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            for metric, values in bootstrap_results.items():
                metric_clean = str(metric).replace("(","_").replace(")","_")
                if auc_suffix:
                    logged_metric = f"{metric_clean}_auc_{aggregation}_{split}{dataloader_idx}"
                else:
                    logged_metric = f"{metric_clean}_{aggregation}_{split}{dataloader_idx}"

                for bootstrap_iteration, value in enumerate(values):
                    writer.writerow({
                        "epoch": self.current_epoch,
                        "split": split,
                        "aggregation": aggregation,
                        "dataloader_idx": dataloader_idx,
                        "bootstrap_iteration": bootstrap_iteration,
                        "metric": metric_clean,
                        "logged_metric": logged_metric,
                        "value": float(value),
                    })

        print(f"Wrote bootstrap iteration results to {output_path}")

    def eval_scores(self, targs, preds, classes=None, bootstrap=False, return_bootstrap_results=False):
        if self.task == "classification_and_regression":
            cls_preds = preds[:, :-35]
            reg_preds = preds[:, -35:]
                
            cls_targs = targs[:, :-35]
            reg_targs = targs[:, -35:]

            cls_metrics = {}
            reg_metrics = {}
        
            _, _, cls_res = multiclass_roc_curve(cls_targs, cls_preds, classes=classes[:-35])
            cls_metrics = {f"{k}": v for k, v in cls_res.items()}

            reg_res = regression_metrics(reg_targs, reg_preds, metrics=["mae"], target_names=classes[-35:])
            reg_metrics = {f"{k}": v for k, v in reg_res.items()}

            cls_metrics["composite_score"] = (1.0 - cls_res["macro"]) + reg_res["mae"]

            if bootstrap:
                bootstrap_iterations = getattr(self.hparams, "bootstrap_iterations", 0)
                cls_point, cls_low, cls_high, _, cls_results = empirical_bootstrap(
                    input_tuple=(cls_targs, cls_preds),
                    score_fn=mcrc_flat,
                    n_iterations=bootstrap_iterations,
                    score_fn_kwargs={"classes": classes[:-35]},
                    return_results=True,
                )
                reg_metric_names = ["mae"] + [
                    f"{target_name}_mae"
                    for target_name in (classes[-35:] if classes is not None else [str(i) for i in range(reg_targs.shape[1])])
                ]
                reg_point, reg_low, reg_high, _, reg_results = empirical_bootstrap(
                    input_tuple=(reg_targs, reg_preds),
                    score_fn=regression_flat,
                    n_iterations=bootstrap_iterations,
                    score_fn_kwargs={"metrics": ["mae"], "target_names": classes[-35:]},
                    return_results=True,
                )

                cls_bootstrap = {}
                for i, k in enumerate(cls_res.keys()):
                    cls_bootstrap[f"{k}"] = cls_point[i]
                    cls_bootstrap[f"{k}_low"] = cls_low[i]
                    cls_bootstrap[f"{k}_high"] = cls_high[i]
                
                reg_bootstrap = {}
                for i, k in enumerate(reg_metric_names):
                    reg_bootstrap[f"{k}"] = reg_point[i]
                    reg_bootstrap[f"{k}_low"] = reg_low[i]
                    reg_bootstrap[f"{k}_high"] = reg_high[i]

                composite_point = (1.0 - cls_bootstrap["macro"]) + reg_bootstrap["mae"]
                composite_low = (1.0 - cls_bootstrap["macro_low"]) + reg_bootstrap["mae_low"]
                composite_high = (1.0 - cls_bootstrap["macro_high"]) + reg_bootstrap["mae_high"]

                cls_bootstrap["composite_score"] = composite_point
                cls_bootstrap["composite_score_low"] = composite_low
                cls_bootstrap["composite_score_high"] = composite_high

                bootstrap_results = {}
                for i, k in enumerate(cls_res.keys()):
                    bootstrap_results[f"{k}"] = cls_results[:, i]
                for i, k in enumerate(reg_metric_names):
                    bootstrap_results[f"{k}"] = reg_results[:, i]
                bootstrap_results["composite_score"] = (1.0 - bootstrap_results["macro"]) + bootstrap_results["mae"]
                
                metrics = {**cls_bootstrap, **reg_bootstrap}
                if return_bootstrap_results:
                    return metrics, bootstrap_results
                return metrics
            else:
                metrics = {**cls_metrics, **reg_metrics}
                if return_bootstrap_results:
                    return metrics, None
                return metrics
        else:
            _,_,res = multiclass_roc_curve(targs,preds,classes=classes)
            if(bootstrap):
                point,low,high,_,bootstrap_array = empirical_bootstrap((targs,preds), mcrc_flat, n_iterations=getattr(self.hparams, "bootstrap_iterations", 0),score_fn_kwargs={"classes":classes}, return_results=True)
                res2={}
                for i,k in enumerate(res.keys()):
                    res2[k]=point[i]
                    res2[k+"_low"]=low[i]
                    res2[k+"_high"]=high[i]
                if return_bootstrap_results:
                    bootstrap_results = {
                        k: bootstrap_array[:, i]
                        for i, k in enumerate(res.keys())
                    }
                    return res2, bootstrap_results
                return res2
            if return_bootstrap_results:
                return res, None
            return res

    def eval_auprc_scores(self, targs, preds, classes=None):
        _, _, res = multiclass_roc_curve(targs, preds, classes=classes, precision_recall=True)
        return res

    def on_valtest_epoch_eval(self, outputs_all, dataloader_idx, test=False):
        preds_all = torch.cat(outputs_all["preds"]).cpu()
        targs_all = torch.cat(outputs_all["targs"]).cpu()

        if self.task != "classification_and_regression":
            if(self.hparams.finetune_dataset == "thew" or self.hparams.finetune_dataset.startswith("segrhythm")):
                preds_all = F.softmax(preds_all.float(),dim=-1)
                targs_all = torch.eye(len(self.lbl_itos))[targs_all].to(preds_all.device) 
            else:
                preds_all = torch.sigmoid(preds_all.float())

        if self.task == "classification_and_regression":
            cls_preds = torch.sigmoid(preds_all[:, :-35]).float()
            reg_preds = preds_all[:, -35:].float()
            preds_all = torch.cat([cls_preds, reg_preds], dim=1)

        preds_all = preds_all.numpy()
        targs_all = targs_all.numpy()

        bootstrap = test and getattr(self.hparams, "bootstrap_iterations", 0) > 0

        # Non-aggregated

        res, bootstrap_results = self.eval_scores(
            targs_all,
            preds_all,
            self.lbl_itos,
            bootstrap=bootstrap,
            return_bootstrap_results=True,
        )
        if self.task == "classification_and_regression":
            res = {k+"_noagg_"+("test" if test else "val")+str(dataloader_idx): v for k, v in res.items()}
            res = {k.replace("(","_").replace(")","_"):v for k,v in res.items()}
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} composite score noagg: {res['composite_score_noagg_'+('test' if test else 'val')+str(dataloader_idx)]}")
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} macro auroc noagg: {res['macro_noagg_'+('test' if test else 'val')+str(dataloader_idx)]}")
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} mae noagg: {res['mae_noagg_'+('test' if test else 'val')+str(dataloader_idx)]}")
            self.write_bootstrap_results(bootstrap_results, "test" if test else "val", "noagg", dataloader_idx, auc_suffix=False)
        else:
            res = {k+"_auc_noagg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in res.items()}
            res = {k.replace("(","_").replace(")","_"):v for k,v in res.items()}
            auprc = self.eval_auprc_scores(targs_all, preds_all, classes=self.lbl_itos)
            auprc = {k+"_auprc_noagg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in auprc.items()}
            auprc = {k.replace("(","_").replace(")","_"):v for k,v in auprc.items()}
            print(
                "epoch", self.current_epoch, "test" if test else "val", "noagg auroc:",
                res["macro_auc_noagg_"+("test" if test else "val")+str(dataloader_idx)],
                "auprc:",
                auprc["macro_auprc_noagg_"+("test" if test else "val")+str(dataloader_idx)],
                flush=True,
            )
            self.write_bootstrap_results(bootstrap_results, "test" if test else "val", "noagg", dataloader_idx, auc_suffix=True)
            self.log_dict(auprc)
        self.log_dict(res)

        # Aggregated
        preds_all_agg,targs_all_agg = self.val_datasets[0].aggregate_predictions(preds_all,targs_all,self.test_idmaps[dataloader_idx] if test else self.val_idmaps[dataloader_idx],aggregate_fn=np.mean)
        res_agg, bootstrap_results_agg = self.eval_scores(
            targs_all_agg,
            preds_all_agg,
            self.lbl_itos,
            bootstrap=bootstrap,
            return_bootstrap_results=True,
        )
        if self.task == "classification_and_regression":
            res_agg = {k+"_agg_"+("test" if test else "val")+str(dataloader_idx): v for k, v in res_agg.items()}
            res_agg = {k.replace("(","_").replace(")","_"):v for k,v in res_agg.items()}
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} composite score agg: {res_agg['composite_score_agg_'+('test' if test else 'val')+str(dataloader_idx)]}")
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} macro auroc agg: {res_agg['macro_agg_'+('test' if test else 'val')+str(dataloader_idx)]}")
            print(f"epoch {self.current_epoch} {'test' if test else 'val'} mae agg: {res_agg['mae_agg_'+('test' if test else 'val')+str(dataloader_idx)]}")
            self.write_bootstrap_results(bootstrap_results_agg, "test" if test else "val", "agg", dataloader_idx, auc_suffix=False)
        else:            
            res_agg = {k+"_auc_agg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in res_agg.items()}
            res_agg = {k.replace("(","_").replace(")","_"):v for k,v in res_agg.items()}            
            auprc_agg = self.eval_auprc_scores(targs_all_agg, preds_all_agg, classes=self.lbl_itos)
            auprc_agg = {k+"_auprc_agg_"+("test" if test else "val")+str(dataloader_idx):v for k,v in auprc_agg.items()}
            auprc_agg = {k.replace("(","_").replace(")","_"):v for k,v in auprc_agg.items()}
            print(
                "epoch", self.current_epoch, "test" if test else "val", "agg auroc:",
                res_agg["macro_auc_agg_"+("test" if test else "val")+str(dataloader_idx)],
                "auprc:",
                auprc_agg["macro_auprc_agg_"+("test" if test else "val")+str(dataloader_idx)],
                flush=True,
            )
            self.write_bootstrap_results(bootstrap_results_agg, "test" if test else "val", "agg", dataloader_idx, auc_suffix=True)
            self.log_dict(auprc_agg)
        self.log_dict(res_agg)
        
        
        if self.hparams.export_predictions:
            # Find version number of the prediction folder

            prediction_dir = Path(self.hparams.prediction_path)
            prediction_dir.mkdir(parents=True, exist_ok=True)

            existing_versions = []
            for folder in prediction_dir.glob(f"{self.hparams.finetune_dataset}_version_*"):
                if folder.is_dir():
                    try:
                        ver_num = int(folder.name.split('_')[-1])
                        existing_versions.append(ver_num)
                    except ValueError:
                        raise ValueError(f"Unexpected folder name format: '{folder.name}'. Expected a suffix with an integer version number.")
            
            current_version = max(existing_versions) + 1 if existing_versions else 0

            if test:
                self._export_predictions(
                    preds=preds_all,
                    targs=targs_all,
                    dataloader_idx=dataloader_idx,
                    epoch=self.current_epoch,
                    agg=False,
                    version=current_version
                )

                self._export_predictions(
                    preds=preds_all_agg,
                    targs=targs_all_agg,
                    dataloader_idx = dataloader_idx,
                    epoch=self.current_epoch,
                    agg=True,
                    version=current_version
                )

    
    def _export_predictions(self, preds, targs, dataloader_idx, epoch, agg, version):
        prediction_dir = Path(self.hparams.prediction_path)        
        version_folder_name = f"{self.hparams.finetune_dataset}_version_{version}"
        full_prediction_dir = prediction_dir / version_folder_name / ("agg" if agg else "noagg")
        full_prediction_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            full_prediction_dir / f"test_{dataloader_idx}_epoch_{epoch}_{'agg' if agg else 'noagg'}.npz",
            preds=preds,
            targs=targs,
            lbl_itos=np.array(self.lbl_itos),
            epoch=epoch
        )    
    
    def setup_dataset(self, target_folder, df_mapped, lbl_itos, mean, std):
        return df_mapped, lbl_itos, mean, std

    def setup_transforms(self, tfms_lst):
        return tfms_lst

    def reset_eval_buffers(self):
        self.val_preds=[[] for _ in range(len(self.val_datasets))]
        self.val_targs=[[] for _ in range(len(self.val_datasets))]
        self.test_preds=[[] for _ in range(len(self.test_datasets))]
        self.test_targs=[[] for _ in range(len(self.test_datasets))]
        self.val_idmaps = [ds.get_id_mapping() for ds in self.val_datasets]
        self.test_idmaps = [ds.get_id_mapping() for ds in self.test_datasets]

    def precompute_embeddings(self, dataset, desc="embeddings"):
        """Precompute and cache embeddings for all samples in a dataset"""
        embeddings_list = []
        labels_list = []
        
        self.model.eval()
        with torch.no_grad():
            for d in tqdm(dataset, total=len(dataset), desc=desc, unit="sample"):
                # Compute embedding
                seq = d.seq.to(self.device)
                emb = self.model.initialize_embeddings(seq)
                
                embeddings_list.append(emb.cpu())
                labels_list.append(d.label)
        
        embeddings = torch.stack(embeddings_list, dim=0)
        labels = torch.stack(labels_list, dim=0)
        
        return embeddings, labels

    def apply_inference_intervals(self, df, split_name):
        interval_csv = getattr(self.hparams, "inference_interval_csv", "")
        if _is_empty_arg(interval_csv):
            return df
        if not _split_requested(split_name, getattr(self.hparams, "inference_interval_splits", "test")):
            return df

        interval_paths = _csv_paths(interval_csv)
        missing_paths = [path for path in interval_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(f"Inference interval CSV not found: {missing_paths[0]}")

        intervals = pd.concat(
            [pd.read_csv(path) for path in interval_paths],
            ignore_index=True,
        )
        split_col = _resolve_column(
            intervals.columns,
            getattr(self.hparams, "inference_interval_split_col", ""),
            ["split"],
        )
        if split_col is not None:
            allowed = _split_values(split_name)
            intervals = intervals[
                intervals[split_col].astype(str).str.lower().isin(allowed)
            ].copy()

        source_key = _resolve_column(
            intervals.columns,
            getattr(self.hparams, "inference_interval_name_col", ""),
            ["record_name", "name", "NAME"],
        )
        target_key = _resolve_column(
            df.columns,
            getattr(self.hparams, "inference_interval_target_col", ""),
            ["record_name", "name", "NAME"],
        )
        if source_key is None or target_key is None:
            raise ValueError(
                "Could not resolve interval merge key. Pass "
                "--inference-interval-name-col and/or --inference-interval-target-col."
            )

        start_sec_col = _resolve_column(
            intervals.columns,
            getattr(self.hparams, "inference_interval_start_sec_col", ""),
            ["start_sec", "start_seconds", "window_start_sec", "window_start", "start"],
        )
        end_sec_col = _resolve_column(
            intervals.columns,
            getattr(self.hparams, "inference_interval_end_sec_col", ""),
            ["end_sec", "end_seconds", "window_end_sec", "window_end", "end"],
        )
        start_idx_col = _resolve_column(
            intervals.columns,
            getattr(self.hparams, "inference_interval_start_idx_col", ""),
            ["start_idx", "start_sample", "start_sample_idx"],
        )
        end_idx_col = _resolve_column(
            intervals.columns,
            getattr(self.hparams, "inference_interval_end_idx_col", ""),
            ["end_idx", "end_sample", "end_sample_idx"],
        )
        anchor_idx_col = _resolve_column(
            intervals.columns,
            getattr(self.hparams, "inference_interval_anchor_idx_col", ""),
            ["abnormal_start", "anchor_idx", "anchor_sample", "anchor_sample_idx", "event_ind"],
        )
        lookback_sec = float(getattr(self.hparams, "inference_interval_lookback_sec", 0) or 0)
        use_anchor_window = lookback_sec > 0
        if use_anchor_window and anchor_idx_col is None:
            raise ValueError(
                "Anchor-window inference needs an anchor sample column. Pass "
                "--inference-interval-anchor-idx-col, or include abnormal_start."
            )
        if not use_anchor_window:
            if start_sec_col is None and start_idx_col is None:
                raise ValueError("Interval CSV needs a start seconds or start index column.")
            if end_sec_col is None and end_idx_col is None:
                raise ValueError("Interval CSV needs an end seconds or end index column.")

        interval_rows = pd.DataFrame({target_key: intervals[source_key].astype(str)})
        sample_fs = float(self.hparams.fs_data)
        if use_anchor_window:
            anchor_idx = pd.to_numeric(intervals[anchor_idx_col], errors="coerce")
            interval_rows["start_idx"] = anchor_idx - lookback_sec * sample_fs
            interval_rows["end_idx"] = anchor_idx
        else:
            if start_idx_col is not None:
                interval_rows["start_idx"] = pd.to_numeric(intervals[start_idx_col], errors="coerce")
            else:
                interval_rows["start_idx"] = pd.to_numeric(intervals[start_sec_col], errors="coerce") * sample_fs
            if end_idx_col is not None:
                interval_rows["end_idx"] = pd.to_numeric(intervals[end_idx_col], errors="coerce")
            else:
                interval_rows["end_idx"] = pd.to_numeric(intervals[end_sec_col], errors="coerce") * sample_fs

        interval_rows = interval_rows.dropna(subset=["start_idx", "end_idx"]).copy()
        interval_rows["start_idx"] = interval_rows["start_idx"].round().astype(np.int64)
        interval_rows["end_idx"] = interval_rows["end_idx"].round().astype(np.int64)
        interval_rows = interval_rows[interval_rows["end_idx"] > interval_rows["start_idx"]]

        base = df.copy()
        base[target_key] = base[target_key].astype(str)
        merged = base.merge(interval_rows, on=target_key, how="inner")
        if merged.empty:
            raise ValueError(
                f"No {split_name} rows matched {interval_csv} using "
                f"{target_key!r} <-> {source_key!r}."
            )
        print(
            f"Applied inference intervals to {split_name}: "
            f"{len(df)} base rows -> {len(merged)} interval rows"
        )
        return merged

    def _labels_as_matrix(self, labels):
        if labels is None:
            return None
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        labels = np.asarray(labels)
        if labels.size == 0:
            return labels.reshape(0, 1)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)
        try:
            return labels.astype(float)
        except (TypeError, ValueError):
            return None

    def _prevalence_text(self, labels):
        labels = self._labels_as_matrix(labels)
        if labels is None or labels.shape[0] == 0:
            return "n/a"

        valid = ~np.isnan(labels)
        counts = valid.sum(axis=0)
        positives = np.nansum(labels, axis=0)
        prevalence = np.divide(
            positives,
            counts,
            out=np.full_like(positives, np.nan, dtype=float),
            where=counts > 0,
        )

        if labels.shape[1] == 1:
            return f"{prevalence[0]:.4f} ({int(round(positives[0]))}/{int(counts[0])})"

        class_names = [str(x) for x in self.lbl_itos] if self.lbl_itos is not None else [str(i) for i in range(labels.shape[1])]
        parts = [
            f"{class_names[i]}={prevalence[i]:.4f} ({int(round(positives[i]))}/{int(counts[i])})"
            for i in range(min(labels.shape[1], 5))
        ]
        if labels.shape[1] > 5:
            parts.append("...")
        return ", ".join(parts)

    def _dataset_label_arrays(self, dataset):
        if isinstance(dataset, ConcatTimeSeriesDataset):
            record_labels = []
            sample_labels = []
            n_records = 0
            n_samples = 0
            for child in dataset.datasets:
                child_records, child_samples, child_record_labels, child_sample_labels = self._dataset_label_arrays(child)
                n_records += child_records
                n_samples += child_samples
                if child_record_labels is not None:
                    record_labels.append(child_record_labels)
                if child_sample_labels is not None:
                    sample_labels.append(child_sample_labels)
            record_labels = np.concatenate(record_labels, axis=0) if record_labels else None
            sample_labels = np.concatenate(sample_labels, axis=0) if sample_labels else None
            return n_records, n_samples, record_labels, sample_labels

        if hasattr(dataset, "timeseries_df_label"):
            record_labels = self._labels_as_matrix(dataset.timeseries_df_label)
            sample_labels = record_labels[dataset.df_idx_mapping] if record_labels is not None else None
            return len(dataset.timeseries_df_label), len(dataset), record_labels, sample_labels

        if hasattr(dataset, "labels"):
            labels = self._labels_as_matrix(dataset.labels)
            return len(dataset), len(dataset), labels, labels

        return 0, len(dataset), None, None

    def print_dataset_summary(self, split_name, dataset):
        n_records, n_samples, record_labels, sample_labels = self._dataset_label_arrays(dataset)
        window_seconds = f"{self.hparams.input_size:g}s"
        print(
            f"{split_name} dataset: {n_samples} {window_seconds} samples from {n_records} rows; "
            f"prevalence ({window_seconds} samples): {self._prevalence_text(sample_labels)}; "
            f"prevalence (rows): {self._prevalence_text(record_labels)}",
            flush=True,
        )

    def setup(self, stage):
        if getattr(self, "embeddings_precomputed", False):
            print(f"setup({stage}): using precomputed embedding datasets")
            self.reset_eval_buffers()
            return

        raw_wfdb = self.hparams.data_backend == "wfdb"
        sample_fs = self.hparams.fs_data
        input_size_data = int(self.hparams.input_size*sample_fs)
        chunkify_train = self.hparams.chunkify_train
        chunk_length_train = int(self.hparams.chunk_length_train*input_size_data) if chunkify_train else 0
        stride_train = int(self.hparams.stride_fraction_train*input_size_data)
        chunkify_valtest = True
        chunk_length_valtest = input_size_data if chunkify_valtest else 0
        stride_valtest = int(self.hparams.stride_fraction_valtest*input_size_data)
        train_datasets = []
        val_datasets = []
        test_datasets = []
        self.ds_mean = None
        self.ds_std = None
        self.lbl_itos = None
        for i,target_folder in enumerate(list(self.hparams.data.split(","))):
            target_folder = Path(target_folder)
            tfms_lst = []
            if raw_wfdb:
                df_train, df_val, df_test, lbl_itos, mean, std = load_wfdb_dataset(str(target_folder), data_fs=self.hparams.fs_data)
            else:
                df_mapped, lbl_itos,  mean, std = load_dataset(target_folder, df_filename="df_memmap.pkl")
                df_mapped, lbl_itos,  mean, std = self.setup_dataset(target_folder, df_mapped, lbl_itos, mean, std)
                print("Folder:",target_folder,"Samples:",len(df_mapped))
                max_fold_id = df_mapped.strat_fold.max()
                df_train = df_mapped[df_mapped.strat_fold<max_fold_id-1]
                df_val = df_mapped[df_mapped.strat_fold==max_fold_id-1]
                df_test = df_mapped[df_mapped.strat_fold==max_fold_id]

            df_train = self.apply_inference_intervals(df_train, "train")
            df_val = self.apply_inference_intervals(df_val, "val")
            df_test = self.apply_inference_intervals(df_test, "test")

            if (self.hparams.fs_model != self.hparams.fs_data):
                tfms_lst.append(Resample(self.hparams.fs_data, self.hparams.fs_model))
            if(self.lbl_itos is None):
                self.lbl_itos = lbl_itos
            if self.ds_mean is None:
                self.ds_mean = mean
                self.ds_std = std
            if self.hparams.normalize:
                tfms_lst.append(Normalize(self.ds_mean, self.ds_std))
            if hasattr(self.model, 'get_model_transforms'):
                tfms_lst = self.model.get_model_transforms(tfms_lst)
            tfms_lst.append(ToTensor())
            tfms = tfms_lst[0] if len(tfms_lst)==1 else transforms.Compose(tfms_lst)

            dataset_config_train = TimeSeriesDatasetConfig(
                df=df_train,
                output_size=input_size_data,
                data_folder=target_folder,
                chunk_length=chunk_length_train,
                min_chunk_length=input_size_data,
                stride=stride_train,
                transforms=tfms,
                col_lbl="label",
                memmap_filename=None if raw_wfdb else target_folder/("memmap.npy"),
                raw_wfdb=raw_wfdb,
                raw_wfdb_target_fs=self.hparams.fs_data,
                raw_wfdb_channels=self.hparams.input_channels,
                raw_wfdb_clip_amp=None)
            dataset_config_train.allow_multiple_keys = not _is_empty_arg(getattr(self.hparams, "inference_interval_csv", ""))
            
            train_datasets.append(TimeSeriesDataset(dataset_config_train))
            dataset_config_val = dataclasses.replace(dataset_config_train)
            dataset_config_val.df = df_val
            dataset_config_val.chunk_length= chunk_length_valtest
            dataset_config_val.stride= stride_valtest
            dataset_config_val.transforms= tfms
            dataset_config_val.allow_multiple_keys = not _is_empty_arg(getattr(self.hparams, "inference_interval_csv", ""))
            val_datasets.append(TimeSeriesDataset(dataset_config_val))
            dataset_config_test = dataclasses.replace(dataset_config_val)
            dataset_config_test.df = df_test
            dataset_config_test.allow_multiple_keys = not _is_empty_arg(getattr(self.hparams, "inference_interval_csv", ""))
            test_datasets.append(TimeSeriesDataset(dataset_config_test))
            print("\n",target_folder)
            if(i<len(self.hparams.data.split(","))):
                self.print_dataset_summary("train", train_datasets[-1])
            self.print_dataset_summary("val", val_datasets[-1])
            self.print_dataset_summary("test", test_datasets[-1])
        if(len(train_datasets)>1):
            print("\nCombined:")
            self.train_dataset = ConcatTimeSeriesDataset(train_datasets)
            self.val_datasets = [ConcatTimeSeriesDataset(val_datasets)]+val_datasets
            self.print_dataset_summary("train total", self.train_dataset)
            self.print_dataset_summary("val total", self.val_datasets[0])
            self.test_datasets = [ConcatTimeSeriesDataset(test_datasets)]+test_datasets
            self.print_dataset_summary("test total", self.test_datasets[0])
        else:
            self.train_dataset = train_datasets[0]
            self.val_datasets = val_datasets
            self.test_datasets = test_datasets

        self.using_precomputed_embeddings = hasattr(self.model, "initialize_embeddings")
        if self.using_precomputed_embeddings:
            print("\nPrecomputing embeddings...")
            
            # Precompute train embeddings
            train_emb, train_lbl = self.precompute_embeddings(self.train_dataset, desc="train embeddings")
            self.train_dataset = EmbeddingDataset(train_emb, train_lbl)
            print(f"Train embeddings: {train_emb.shape}")
            
            # Precompute val embeddings
            val_embedding_datasets = []
            for i, val_ds in enumerate(self.val_datasets):
                val_emb, val_lbl = self.precompute_embeddings(val_ds, desc=f"val embeddings {i}")
                val_embedding_datasets.append(EmbeddingDataset(val_emb, val_lbl))
                print(f"Val dataset {i} embeddings: {val_emb.shape}")
            self.val_datasets = val_embedding_datasets
            
            # Precompute test embeddings
            test_embedding_datasets = []
            for i, test_ds in enumerate(self.test_datasets):
                test_emb, test_lbl = self.precompute_embeddings(test_ds, desc=f"test embeddings {i}")
                test_embedding_datasets.append(EmbeddingDataset(test_emb, test_lbl))
                print(f"Test dataset {i} embeddings: {test_emb.shape}")
            self.test_datasets = test_embedding_datasets

            self.embeddings_precomputed = True
        self.reset_eval_buffers()

    def embedding_collate_fn(self, batch):
        embeddings = torch.stack([b['embedding'] for b in batch])
        labels = torch.stack([b['label'] for b in batch])
        return {"embedding": embeddings, "label": labels}
    def dataloader_kwargs(self, shuffle=False, drop_last=False):
        num_workers = int(getattr(self.hparams, "num_workers", 0) or 0)
        kwargs = {
            "batch_size": self.hparams.batch_size,
            "collate_fn": self.embedding_collate_fn if getattr(self, "using_precomputed_embeddings", False) else tsdata_collate_fn,
            "num_workers": num_workers,
            "shuffle": shuffle,
            "drop_last": drop_last,
        }
        if num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs
    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_kwargs(shuffle=True, drop_last=True))
    def val_dataloader(self):
        return [DataLoader(ds, **self.dataloader_kwargs()) for ds in self.val_datasets]
    def test_dataloader(self):
        return [DataLoader(ds, **self.dataloader_kwargs()) for ds in self.test_datasets]

    def _step(self,data_batch, batch_idx, train, test=False, dataloader_idx=0):
        model_input = data_batch["embedding"] if "embedding" in data_batch else data_batch["seq"]
        preds_all = self.forward(model_input)
        loss = self.criterion(preds_all, data_batch["label"].float())
        
        if(not train and not test):
            self.val_preds[dataloader_idx].append(preds_all.detach())
            self.val_targs[dataloader_idx].append(data_batch["label"])
        elif(not train and test):
            self.test_preds[dataloader_idx].append(preds_all.detach())
            self.test_targs[dataloader_idx].append(data_batch["label"])
        return loss
    # def _step(self, data_batch, batch_idx, train, test=False, dataloader_idx=0):
    #     preds_all = self.forward(data_batch["seq"])
    #     targets = data_batch["label"]

    #     print(f"Step Pred shape: {preds_all.shape}")
    #     print(f"Step Targ shape: {targets.shape}")
        
    #     nan_mask = torch.isnan(targets)
    #     special_value_mask = (targets == -999)
        
    #     invalid_mask = nan_mask | special_value_mask
        
    #     if torch.any(invalid_mask):
    #         row_has_invalid = invalid_mask.any(dim=1)
    #         completely_valid_rows = ~row_has_invalid
            
    #         targets_clean = targets[completely_valid_rows]
    #         preds_clean = preds_all[completely_valid_rows]
    #         loss = self.criterion(preds_clean, targets_clean.float())
    #     else:
    #         loss = self.criterion(preds_all, targets.float())
        
    #     self.log("train_loss" if train else ("test_loss" if test else "val_loss"), loss)
        
    #     if(not train and not test):
    #         self.val_preds[dataloader_idx].append(preds_all.detach())
    #         self.val_targs[dataloader_idx].append(data_batch["label"])
    #     elif(not train and test):
    #         self.test_preds[dataloader_idx].append(preds_all.detach())
    #         self.test_targs[dataloader_idx].append(data_batch["label"])
    #     return loss
    def training_step(self, train_batch, batch_idx):
        return self._step(train_batch,batch_idx,train=True)
    def validation_step(self, val_batch, batch_idx, dataloader_idx=0):
        return self._step(val_batch,batch_idx,train=False,test=False, dataloader_idx=dataloader_idx)
    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        return self._step(test_batch,batch_idx,train=False,test=True, dataloader_idx=dataloader_idx)
    def configure_optimizers(self):
        if(self.hparams.optimizer == "sgd"):
            opt = torch.optim.SGD
        elif(self.hparams.optimizer == "adam"):
            opt = torch.optim.AdamW
        else:
            raise NotImplementedError("Unknown Optimizer.")
        params = self.parameters()
        optimizer = opt(params, self.lr, weight_decay=self.hparams.weight_decay)
        if(self.hparams.lr_schedule=="const"):
            scheduler = get_constant_schedule(optimizer)
        elif(self.hparams.lr_schedule=="warmup-const"):
            scheduler = get_constant_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif(self.hparams.lr_schedule=="warmup-cos"):
            scheduler = get_cosine_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=0.5)
        elif(self.hparams.lr_schedule=="warmup-cos-restart"):
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)
        elif(self.hparams.lr_schedule=="warmup-poly"):
            scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps,self.hparams.epochs*len(self.train_dataloader()),num_cycles=self.hparams.epochs-1)   
        elif(self.hparams.lr_schedule=="warmup-invsqrt"):
            scheduler = get_invsqrt_decay_schedule_with_warmup(optimizer,self.hparams.lr_num_warmup_steps)
        elif(self.hparams.lr_schedule=="linear"):
            scheduler = get_linear_schedule_with_warmup(optimizer, 0, self.hparams.epochs*len(self.train_dataloader()))
        else:
            assert(False)
        return (
        [optimizer],
        [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        ])
    def load_weights_from_checkpoint(self, checkpoint):
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage,)
        pretrained_dict = checkpoint["state_dict"]
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
    def load_state_dict(self, state_dict, strict=True):
        for name, param in self.named_parameters():
            if name in state_dict:
                param.data = state_dict[name].data.to(param.device)
            elif strict:
                raise KeyError(f"Key {name} not found in state_dict")
        for name, param in self.named_buffers():
            if name in state_dict:
                param.data = state_dict[name].data.to(param.device)
            elif strict:
                raise KeyError(f"Buffer {name} not found in state_dict") 
