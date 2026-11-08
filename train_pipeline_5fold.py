# -*- coding: utf-8 -*-
r"""
Final retained school-building multisource 5-fold pipeline
==========================================================

ONLY the final retained technical route is included:

    1) SegFormer pure visual baseline
    2) empirical-radius school POI + SegFormer decoder-feature adaptive lambda(x)
    3) scalar-input VLM scene-triggered recovery
    4) active-AOI outside soft suppression

Explicitly NOT included:
    - valid/invalid school-POI filtering
    - competitive non-school POI
    - fixed-lambda POI as a reported final stage
    - visual-feature VLM recovery (tested and discarded)
    - AOI positive recovery

This is a STAGE-WISE pipeline, not end-to-end joint training.
Every downstream stage freezes all upstream stages.

Formal 5-fold rules
-------------------
For every fold independently:
    - SegFormer is trained on this fold's TRAIN and selected by VAL IoU.
    - POI radii are estimated from this fold's TRAIN buildings only.
    - lambda_ref is selected on this fold's TRAIN only.
    - Adaptive-POI Gate is trained on TRAIN; best epoch selected on VAL.
    - Scalar VLM recovery Gate is trained on TRAIN; best epoch selected on VAL.
    - AOI sigma/floor are selected on VAL only.
    - Then the TEST split is processed once through the frozen full chain.

Process TEST metrics are reported for all four retained stages:
    SegFormer
    Adaptive visual-feature POI
    Adaptive POI + VLM
    Adaptive POI + VLM + AOI

For every stage, metrics are reported for:
    Overall
    小学
    中学
    高等教育

IMPORTANT
---------
The default mode is formal_5fold. In this mode Fold 0 is NOT required to equal
exactly the exploratory Fold-0 numbers from today, because the formal code
re-estimates the three empirical POI radii from Fold-0 TRAIN buildings only.
The qualitative trend can be compared, but exact equality is not expected.

If you first want to verify that the integrated code preserves today's Fold-0
implementation, set:
    PIPELINE_MODE = "reproduce_today_fold0"
This uses the diagnostic radii 123 / 281 / 345 m and only runs Fold 0.

Run in the larse environment:
    conda activate larse
    python run_final_school_multisource_5fold.py

The DGTRS/LRSCLIP scene maps are reused from the already-generated cache.
DGTRS itself is not run by this integrated pipeline.
No rasterio is used.
"""

from __future__ import print_function

import os
import sys
import json
import math
import random
import shutil
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


# =============================================================================
# 0. LOCAL FINAL-ROUTE MODULES
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def import_local(name, filename):
    p = SCRIPT_DIR / filename
    if not p.exists():
        raise FileNotFoundError(
            "Required local module is missing:\n{}".format(p)
        )
    spec = importlib.util.spec_from_file_location(name, str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import {}".format(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SEG = import_local(
    "final_segformer_baseline",
    "train_segformer.py",
)

AP = import_local(
    "final_adaptive_poi",
    "train_poi_gate.py",
)

VLM = import_local(
    "final_scalar_vlm",
    "train_vlm_recovery.py",
)

AOI = import_local(
    "final_aoi",
    "apply_aoi_soft_correction.py",
)

# Force AOI to use exactly this already-imported retained scalar-VLM module.
AOI.REC = VLM


# =============================================================================
# 1. USER CONTROL
# =============================================================================

PIPELINE_MODE = "formal_5fold"
# Options:
#   "formal_5fold"
#   "reproduce_today_fold0"

RUN_FOLDS = (
    [0, 1, 2, 3, 4]
    if PIPELINE_MODE == "formal_5fold"
    else [0]
)

SKIP_COMPLETED_FOLDS = True
SKIP_COMPLETED_STAGES = True

# Output of this integrated experiment.
PIPELINE_ROOT = SCRIPT_DIR / "outputs" / "pipeline_5fold"

# Existing prepared dataset and static sources.
DATASET_ROOT = SCRIPT_DIR / "data"

BASE_SAMPLE_DIR = SCRIPT_DIR / "data" / "auxiliary"

BUILDING_SHP = BASE_SAMPLE_DIR / "school_buildings_merged.shp"
CORE_SHP = BASE_SAMPLE_DIR / "school_cores_merged.shp"
AOI_SHP = BASE_SAMPLE_DIR / "学校AOI.shp"
CORE_MATCH_AUDIT = DATASET_ROOT / "metadata.csv"

# Building-level R95 values produced by the previous descriptive analysis.
# These distances depend only on building geometry + raw same-type school POIs.
# Formal folds use only rows belonging to that fold's TRAIN cores.
BUILDING_R95_CSV = (
    SCRIPT_DIR / "work" / "poi_distances" / "building_poi_influence_R95.csv"
)

# All raw school POI base cache. No valid/invalid POI filtering.
ALL_SCHOOL_POI_CACHE = SCRIPT_DIR / "work" / "poi_priors" / "cache"

# Existing DGTRS/LRSCLIP scene maps from the current 421-tile dataset.
# The three old split folders together cover all current images.

UNIFIED_SCENE_CACHE = SCRIPT_DIR / "work" / "vlm_scene_maps"
# Diagnostic Fold-0 radii used in today's exploratory run.
TODAY_RADII_M = {
    "primary": 123.0,
    "middle": 281.0,
    "higher": 345.0,
}

# Formal empirical-radius definition:
# per-building R95, then school-type P90 across TRAIN buildings.
RADIUS_PERCENTILE = 90.0

# Fixed-lambda TRAIN-only search used only to initialize / scale the final
# adaptive lambda(x) Gate. It is not a reported final stage.
LAMBDA_CANDIDATES = np.round(
    np.arange(0.0, 4.0 + 1e-9, 0.10),
    2,
).tolist()

STOP_IF_LAMBDA_HITS_UPPER_BOUNDARY = True

SEED = 42
THRESHOLD = 0.50
IGNORE_VALUE = 255
PROB_EPS = 1e-6

TYPE_KEYS = ["primary", "middle", "higher"]
TYPE_VALUE = {
    "primary": 1,
    "middle": 2,
    "higher": 3,
}
TYPE_CN = {
    "primary": "小学",
    "middle": "中学",
    "higher": "高等教育",
}
TYPE_CN_TO_KEY = {
    "小学": "primary",
    "中学": "middle",
    "高等教育": "higher",
}

STAGE_SEG = "SegFormer"
STAGE_POI = "Adaptive visual-feature POI"
STAGE_VLM = "Adaptive POI + VLM"
STAGE_FINAL = "Adaptive POI + VLM + AOI"
STAGES = [STAGE_SEG, STAGE_POI, STAGE_VLM, STAGE_FINAL]


# =============================================================================
# 2. BASIC HELPERS
# =============================================================================


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_torch_load(path, map_location="cpu"):
    """Compatibility with both old and new PyTorch; no weights_only dependency."""
    try:
        return torch.load(str(path), map_location=map_location)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def read_ids(path):
    with open(str(path), "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path):
    with open(str(path), "r", encoding="utf-8") as f:
        return json.load(f)


def fold_root(fold_id):
    p = PIPELINE_ROOT / "fold_{}".format(int(fold_id))
    p.mkdir(parents=True, exist_ok=True)
    return p


def prob_path(root, gid):
    root = Path(root)
    p1 = root / (str(gid) + "_school_probability.npy")
    p2 = root / (str(gid) + ".npy")
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    return p1


def read_prob(root, gid):
    p = prob_path(root, gid)
    if not p.exists():
        raise FileNotFoundError(p)
    a = np.load(str(p)).astype(np.float32)
    if a.ndim != 2:
        raise RuntimeError("Probability must be HxW: {}".format(p))
    if not np.isfinite(a).all():
        raise RuntimeError("NaN/Inf in probability: {}".format(p))
    return np.clip(a, 0.0, 1.0)


def save_prob(root, gid, arr):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.save(
        str(root / (str(gid) + "_school_probability.npy")),
        np.asarray(arr, dtype=np.float32),
    )


def prob_cache_complete(root, ids):
    return all(prob_path(root, gid).exists() for gid in ids)


def logit_np(p):
    p = np.clip(p, PROB_EPS, 1.0 - PROB_EPS)
    return np.log(p / (1.0 - p))


def sigmoid_np(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out.astype(np.float32)


def binary_counts(gt, pred):
    valid = gt != IGNORE_VALUE
    g = gt[valid]
    p = pred[valid]
    tp = int(np.sum((g == 1) & (p == 1)))
    tn = int(np.sum((g == 0) & (p == 0)))
    fp = int(np.sum((g == 0) & (p == 1)))
    fn = int(np.sum((g == 1) & (p == 0)))
    return [tp, tn, fp, fn]


def add_counts(total, c):
    for i in range(4):
        total[i] += int(c[i])


def metrics(c):
    tp, tn, fp, fn = [int(x) for x in c]
    eps = 1e-12
    return {
        "precision": float(tp / (tp + fp + eps)),
        "recall": float(tp / (tp + fn + eps)),
        "f1": float(2.0 * tp / (2.0 * tp + fp + fn + eps)),
        "iou": float(tp / (tp + fp + fn + eps)),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def type_counts(type_gt, pred, target_value):
    if not np.any(type_gt == int(target_value)):
        return None
    valid = (type_gt == 0) | (type_gt == int(target_value))
    g = type_gt[valid] == int(target_value)
    p = pred[valid] == 1
    tp = int(np.sum(g & p))
    tn = int(np.sum((~g) & (~p)))
    fp = int(np.sum((~g) & p))
    fn = int(np.sum(g & (~p)))
    return [tp, tn, fp, fn]


def evaluate_probability_dir(ids, prob_dir):
    overall = [0, 0, 0, 0]
    type_total = {k: [0, 0, 0, 0] for k in TYPE_KEYS}
    type_tiles = {k: 0 for k in TYPE_KEYS}
    type_pixels = {k: 0 for k in TYPE_KEYS}
    total_school_pixels = 0

    for gid in ids:
        gt = AP.read_mask(gid)
        type_gt = AP.read_type_mask(gid)
        p = read_prob(prob_dir, gid)
        if not (gt.shape == type_gt.shape == p.shape):
            raise RuntimeError("Metric shape mismatch: {}".format(gid))
        pred = (p >= THRESHOLD).astype(np.uint8)
        add_counts(overall, binary_counts(gt, pred))
        total_school_pixels += int(np.sum(gt == 1))

        for key in TYPE_KEYS:
            tv = TYPE_VALUE[key]
            c = type_counts(type_gt, pred, tv)
            if c is None:
                continue
            add_counts(type_total[key], c)
            type_tiles[key] += 1
            type_pixels[key] += int(np.sum(type_gt == tv))

    out = {
        "Overall": {
            **metrics(overall),
            "positive_tiles": int(len(ids)),
            "gt_target_pixels": int(total_school_pixels),
        }
    }
    for key in TYPE_KEYS:
        out[TYPE_CN[key]] = {
            **metrics(type_total[key]),
            "positive_tiles": int(type_tiles[key]),
            "gt_target_pixels": int(type_pixels[key]),
        }
    return out


def stage_rows(fold_id, stage, result):
    rows = []
    for scope in ["Overall", "小学", "中学", "高等教育"]:
        m = result[scope]
        rows.append({
            "fold": int(fold_id),
            "stage": stage,
            "scope": scope,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "iou": m["iou"],
            "accuracy": m["accuracy"],
            "TP": m["TP"],
            "TN": m["TN"],
            "FP": m["FP"],
            "FN": m["FN"],
            "positive_tiles": m.get("positive_tiles", ""),
            "gt_target_pixels": m.get("gt_target_pixels", ""),
        })
    return rows


# =============================================================================
# 3. ONE-TIME SCENE CACHE UNIFICATION
# =============================================================================

def prepare_unified_scene_cache():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal training pipeline.")

    from prepare_vlm_scene_maps import prepare_scene_maps

    prepare_scene_maps(
        vlm_module=VLM,
        dataset_root=DATASET_ROOT,
        cache_dir=UNIFIED_SCENE_CACHE,
    )


# =============================================================================
# 4. FORMAL TRAIN-ONLY POI RADII
# =============================================================================

_BUILDING_STATS_CACHE = None
_BUILDINGS_CACHE = None
_CORES_CACHE = None
_AUDIT_CACHE = None


def load_radius_sources():
    global _BUILDING_STATS_CACHE
    global _BUILDINGS_CACHE
    global _CORES_CACHE
    global _AUDIT_CACHE

    if _BUILDING_STATS_CACHE is not None:
        return

    for p in [BUILDING_R95_CSV, BUILDING_SHP, CORE_SHP, CORE_MATCH_AUDIT]:
        if not p.exists():
            raise FileNotFoundError(
                "TRAIN-only radius source missing:\n{}".format(p)
            )

    stats = pd.read_csv(BUILDING_R95_CSV, encoding="utf-8-sig")
    needed = {"building_id", "school_type", "distance_p95_m"}
    if not needed.issubset(set(stats.columns)):
        raise RuntimeError(
            "building_poi_influence_R95.csv missing fields: {}"
            .format(sorted(needed - set(stats.columns)))
        )

    buildings = gpd.read_file(str(BUILDING_SHP)).reset_index(drop=False)
    buildings = buildings.rename(columns={"index": "building_id"})
    buildings["building_id"] = buildings["building_id"].astype(int)

    cores = gpd.read_file(str(CORE_SHP)).reset_index(drop=False)
    cores = cores.rename(columns={"index": "core_index"})
    cores["core_index_str"] = cores["core_index"].astype(str)

    audit = pd.read_csv(CORE_MATCH_AUDIT, encoding="utf-8-sig")
    audit["grid_id"] = audit["grid_id"].astype(str)
    audit["core_index"] = audit["core_index"].apply(lambda x: str(int(x)) if isinstance(x, (int, np.integer, float, np.floating)) and float(x).is_integer() else str(x))

    if buildings.crs is None or cores.crs is None:
        raise RuntimeError("Building/core vector CRS is missing.")

    _BUILDING_STATS_CACHE = stats
    _BUILDINGS_CACHE = buildings
    _CORES_CACHE = cores
    _AUDIT_CACHE = audit


def estimate_train_radii(fold_id, train_ids):
    if PIPELINE_MODE == "reproduce_today_fold0":
        return {
            "radii_m": dict(TODAY_RADII_M),
            "raw_p90_m": dict(TODAY_RADII_M),
            "n_train_buildings": {k: None for k in TYPE_KEYS},
            "audit_rows": None,
            "source": "diagnostic fixed 123/281/345 m",
        }

    load_radius_sources()

    train_set = set(str(x) for x in train_ids)
    audit = _AUDIT_CACHE[_AUDIT_CACHE["grid_id"].isin(train_set)].copy()
    train_core_indices = set(audit["core_index"].astype(str).tolist())

    train_cores = _CORES_CACHE[
        _CORES_CACHE["core_index_str"].isin(train_core_indices)
    ].copy()

    if train_cores.empty:
        raise RuntimeError("Fold {} has no TRAIN cores.".format(fold_id))

    train_cores_b = train_cores.to_crs(_BUILDINGS_CACHE.crs)
    try:
        train_union = train_cores_b.geometry.union_all()
    except Exception:
        train_union = train_cores_b.geometry.unary_union

    b = _BUILDINGS_CACHE.copy()
    reps = b.geometry.representative_point()
    in_train = reps.intersects(train_union)
    train_b = b[in_train].copy()

    merged = train_b[["building_id"]].merge(
        _BUILDING_STATS_CACHE,
        on="building_id",
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise RuntimeError("No TRAIN building R95 rows for fold {}.".format(fold_id))

    radii = {}
    raw = {}
    n = {}

    for cn, key in TYPE_CN_TO_KEY.items():
        x = merged.loc[
            merged["school_type"].astype(str) == cn,
            "distance_p95_m",
        ].to_numpy(dtype=np.float64)

        if x.size == 0:
            raise RuntimeError(
                "Fold {} has zero TRAIN building R95 rows for {}."
                .format(fold_id, cn)
            )

        p90 = float(np.percentile(x, RADIUS_PERCENTILE))
        radii[key] = float(int(np.rint(p90)))
        raw[key] = p90
        n[key] = int(x.size)

    out_csv = fold_root(fold_id) / "train_buildings_for_poi_radius.csv"
    merged.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return {
        "radii_m": radii,
        "raw_p90_m": raw,
        "n_train_buildings": n,
        "audit_rows": str(out_csv),
        "source": "Fold TRAIN buildings only; class radius = P90 of per-building R95",
    }


# =============================================================================
# 5. POI PRIOR + TRAIN-ONLY lambda_ref SEARCH
# =============================================================================


def configure_adaptive_common(fold_id, radii_m, lambda_ref, seg_train_dir):
    fold_dir = DATASET_ROOT / "folds" / "fold_{}".format(fold_id)
    seg_root = SEG.OUT_BASE_ROOT / "fold_{}".format(fold_id)

    AP.DATASET_ROOT = DATASET_ROOT
    AP.IMAGE_DIR = DATASET_ROOT / "images"
    AP.MASK_DIR = DATASET_ROOT / "masks_png"
    AP.TYPE_MASK_DIR = DATASET_ROOT / "masks_type_png"
    AP.FOLD_DIR = fold_dir
    AP.SEGFORMER_FOLD_ROOT = seg_root
    AP.SEGFORMER_BEST_CKPT = seg_root / "checkpoints" / "best_model.pth"
    AP.TRAIN_SEG_PROB_DIR = Path(seg_train_dir)
    AP.VAL_SEG_PROB_DIR = seg_root / "val_predictions_best" / "school_probability_npy"
    AP.TEST_SEG_PROB_DIR = seg_root / "test_predictions" / "school_probability_npy"
    AP.ALL_SCHOOL_CACHE_DIR = ALL_SCHOOL_POI_CACHE
    AP.EMPIRICAL_RADIUS_M = dict(radii_m)
    AP.LAMBDA_REF = float(lambda_ref)


def school_prior(gid, radii_m):
    # Reuse the final adaptive-POI implementation exactly.
    old = AP.EMPIRICAL_RADIUS_M
    AP.EMPIRICAL_RADIUS_M = dict(radii_m)
    try:
        return AP.empirical_school_prior(gid)
    finally:
        AP.EMPIRICAL_RADIUS_M = old


def search_lambda_train(fold_id, train_ids, seg_train_dir, radii_m):
    rows = []
    totals = {float(x): [0, 0, 0, 0] for x in LAMBDA_CANDIDATES}
    seg_total = [0, 0, 0, 0]

    print("\nTRAIN-only lambda_ref search...")

    for i, gid in enumerate(train_ids, 1):
        gt = AP.read_mask(gid)
        p_seg = read_prob(seg_train_dir, gid)
        s_poi = school_prior(gid, radii_m)
        pred_seg = (p_seg >= THRESHOLD).astype(np.uint8)
        add_counts(seg_total, binary_counts(gt, pred_seg))
        z = logit_np(p_seg)

        for lam in LAMBDA_CANDIDATES:
            p = sigmoid_np(z + float(lam) * s_poi)
            add_counts(
                totals[float(lam)],
                binary_counts(gt, (p >= THRESHOLD).astype(np.uint8)),
            )

        if i % 25 == 0 or i == len(train_ids):
            print("  lambda [{}/{}] {}".format(i, len(train_ids), gid))

    seg_m = metrics(seg_total)
    for lam in LAMBDA_CANDIDATES:
        m = metrics(totals[float(lam)])
        rows.append({
            "lambda": float(lam),
            **m,
            "delta_precision_vs_seg": m["precision"] - seg_m["precision"],
            "delta_recall_vs_seg": m["recall"] - seg_m["recall"],
            "delta_f1_vs_seg": m["f1"] - seg_m["f1"],
            "delta_iou_vs_seg": m["iou"] - seg_m["iou"],
        })

    best = max(
        rows,
        key=lambda r: (
            float(r["iou"]),
            float(r["f1"]),
            float(r["precision"]),
            -float(r["lambda"]),
        ),
    )

    pd.DataFrame(rows).to_csv(
        fold_root(fold_id) / "lambda_train_search.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if (
        STOP_IF_LAMBDA_HITS_UPPER_BOUNDARY
        and abs(float(best["lambda"]) - float(max(LAMBDA_CANDIDATES))) < 1e-12
    ):
        raise RuntimeError(
            "Fold {} lambda_ref selected the upper boundary {:.2f}; widen the TRAIN search before VAL/TEST."
            .format(fold_id, best["lambda"])
        )

    print(
        "Selected lambda_ref={:.2f} | TRAIN P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(
            best["lambda"],
            best["precision"],
            best["recall"],
            best["f1"],
            best["iou"],
        )
    )

    return float(best["lambda"])


# =============================================================================
# 6. SEGFORMER LOAD / TRAIN-PROBABILITY EXPORT
# =============================================================================


def load_best_segformer(fold_id, device):
    seg_root = SEG.OUT_BASE_ROOT / "fold_{}".format(fold_id)
    ckpt = seg_root / "checkpoints" / "best_model.pth"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    model = SEG.build_model(device)
    obj = safe_torch_load(ckpt, map_location=device)
    model.load_state_dict(obj["model_state_dict"], strict=True)
    model.eval()
    return model, int(obj["epoch"])


@torch.no_grad()
def export_seg_probabilities(model, ids, out_dir, device, label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = [gid for gid in ids if not prob_path(out_dir, gid).exists()]
    if not missing:
        print("{} SegFormer probability cache already complete.".format(label))
        return

    for i, gid in enumerate(missing, 1):
        image_path = SEG.find_file(SEG.IMAGE_DIR, gid, [".tif", ".tiff"])
        rgb = SEG.read_rgb(image_path)
        _, p = SEG.predict_full_image(model, rgb, device, print_windows=False)
        save_prob(out_dir, gid, p)
        if i % 10 == 0 or i == len(missing):
            print("  export Seg {} [{}/{}] {}".format(label, i, len(missing), gid))


# =============================================================================
# 7. FINAL ADAPTIVE-POI GATE
# =============================================================================


def adaptive_stage_paths(fold_id):
    root = fold_root(fold_id) / "adaptive_poi"
    return {
        "root": root,
        "ckpt": root / "checkpoints" / "best_gate.pth",
        "train": root / "train_probability_npy",
        "val": root / "val_probability_npy",
        "test": root / "test_probability_npy",
        "summary": root / "summary.json",
        "log": root / "training_log.csv",
    }


@torch.no_grad()
def export_adaptive_probabilities(model, ids, seg_prob_dir, radii_m, out_dir, device, label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, gid in enumerate(ids, 1):
        image_path = AP.find_file(AP.IMAGE_DIR, gid, [".tif", ".tiff", ".png", ".jpg", ".jpeg"])
        rgb = AP.read_rgb(gid)
        p_seg = read_prob(seg_prob_dir, gid)
        s_poi = school_prior(gid, radii_m)
        lm = AP.predict_lambda_map_full(model, rgb, p_seg, s_poi, device)
        p = AP.adaptive_probability_from_lambda(p_seg, s_poi, lm)
        save_prob(out_dir, gid, p)
        if i % 10 == 0 or i == len(ids):
            print("  adaptive {} [{}/{}] {}".format(label, i, len(ids), gid))


def train_adaptive_poi(
    fold_id,
    seg_model,
    train_ids,
    val_ids,
    seg_train_dir,
    seg_val_dir,
    radii_m,
    lambda_ref,
    device,
):
    paths = adaptive_stage_paths(fold_id)
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["ckpt"].parent.mkdir(parents=True, exist_ok=True)

    if (
        SKIP_COMPLETED_STAGES
        and paths["summary"].exists()
        and paths["ckpt"].exists()
        and prob_cache_complete(paths["train"], train_ids)
        and prob_cache_complete(paths["val"], val_ids)
    ):
        s = read_json(paths["summary"])
        print("[Fold {}] Adaptive POI already complete; skipping training.".format(fold_id))
        return paths, int(s["best_epoch"]), s["best_val_metrics"]

    configure_adaptive_common(
        fold_id=fold_id,
        radii_m=radii_m,
        lambda_ref=lambda_ref,
        seg_train_dir=seg_train_dir,
    )

    # Rare no-op case: TRAIN says lambda_ref=0.
    if lambda_ref <= 1e-12:
        for ids, src, dst in [
            (train_ids, seg_train_dir, paths["train"]),
            (val_ids, seg_val_dir, paths["val"]),
        ]:
            dst.mkdir(parents=True, exist_ok=True)
            for gid in ids:
                save_prob(dst, gid, read_prob(src, gid))

        val_m = evaluate_probability_dir(val_ids, paths["val"])["Overall"]
        write_json(
            paths["summary"],
            {
                "fold_id": int(fold_id),
                "best_epoch": 0,
                "best_val_metrics": val_m,
                "lambda_ref": float(lambda_ref),
                "no_op": True,
            },
        )
        return paths, 0, val_m

    set_seed(SEED)

    model = AP.FrozenSegVisualLambdaNet(
        segformer=seg_model,
        lambda_ref=lambda_ref,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.gate.parameters(),
        lr=AP.GATE_LR,
        weight_decay=AP.WEIGHT_DECAY,
    )

    ds = AP.GateCropDataset(train_ids)
    loader = DataLoader(
        ds,
        batch_size=AP.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    # Epoch 0: exact fixed-lambda reference. It is allowed to remain best.
    val0 = AP.evaluate_split(
        model=model,
        ids=val_ids,
        seg_prob_dir=seg_val_dir,
        device=device,
        save_test=False,
    )
    best_m = dict(val0["metrics"]["Adaptive visual-feature POI"])
    best_epoch = 0

    torch.save(
        {
            "epoch": 0,
            "gate_state_dict": model.gate.state_dict(),
            "best_val_metrics": best_m,
            "lambda_ref": float(lambda_ref),
            "radii_m": dict(radii_m),
        },
        str(paths["ckpt"]),
    )

    rows = [{
        "epoch": 0,
        "train_loss": "",
        "val_precision": best_m["precision"],
        "val_recall": best_m["recall"],
        "val_f1": best_m["f1"],
        "val_iou": best_m["iou"],
    }]

    print(
        "Adaptive POI epoch 00 | Val P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(best_m["precision"], best_m["recall"], best_m["f1"], best_m["iou"])
    )

    for epoch in range(1, AP.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        n_batch = 0

        for x, y, pseg_t, poi_t in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pseg_t = pseg_t.to(device, non_blocking=True)
            poi_t = poi_t.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(pixel_values=x, p_seg=pseg_t, s_poi=poi_t)

            loss_final = AP.probability_loss(out["p_final"], y)
            loss_pref, _, _ = AP.balanced_gate_preference_loss(
                out["gate"], poi_t, y
            )
            loss = loss_final + AP.LAMBDA_GATE_PREF * loss_pref

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite Adaptive-POI loss.")

            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batch += 1

        row = {
            "epoch": int(epoch),
            "train_loss": float(total_loss / max(1, n_batch)),
            "val_precision": "",
            "val_recall": "",
            "val_f1": "",
            "val_iou": "",
        }

        do_val = (epoch % AP.VAL_EVERY == 0) or (epoch == AP.EPOCHS)
        if do_val:
            vr = AP.evaluate_split(
                model=model,
                ids=val_ids,
                seg_prob_dir=seg_val_dir,
                device=device,
                save_test=False,
            )
            vm = vr["metrics"]["Adaptive visual-feature POI"]
            row.update({
                "val_precision": vm["precision"],
                "val_recall": vm["recall"],
                "val_f1": vm["f1"],
                "val_iou": vm["iou"],
            })

            print(
                "Adaptive POI epoch {:02d} | Val P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
                .format(epoch, vm["precision"], vm["recall"], vm["f1"], vm["iou"])
            )

            if AP.better_metric(vm, best_m):
                best_m = dict(vm)
                best_epoch = int(epoch)
                torch.save(
                    {
                        "epoch": int(epoch),
                        "gate_state_dict": model.gate.state_dict(),
                        "best_val_metrics": best_m,
                        "lambda_ref": float(lambda_ref),
                        "radii_m": dict(radii_m),
                    },
                    str(paths["ckpt"]),
                )
                print("  -> new best Adaptive-POI Gate")

        rows.append(row)
        pd.DataFrame(rows).to_csv(
            paths["log"], index=False, encoding="utf-8-sig"
        )

    best_obj = safe_torch_load(paths["ckpt"], map_location=device)
    model.gate.load_state_dict(best_obj["gate_state_dict"], strict=True)
    model.eval()

    print(
        "Selected Adaptive-POI epoch {} | Val IoU={:.4f}"
        .format(best_epoch, best_m["iou"])
    )

    export_adaptive_probabilities(
        model, train_ids, seg_train_dir, radii_m, paths["train"], device, "TRAIN"
    )
    export_adaptive_probabilities(
        model, val_ids, seg_val_dir, radii_m, paths["val"], device, "VAL"
    )

    write_json(
        paths["summary"],
        {
            "fold_id": int(fold_id),
            "best_epoch": int(best_epoch),
            "best_val_metrics": best_m,
            "lambda_ref": float(lambda_ref),
            "radii_m": {k: float(v) for k, v in radii_m.items()},
            "segformer_frozen": True,
            "method": "empirical-radius POI + SegFormer decoder-feature adaptive lambda(x)",
        },
    )

    return paths, best_epoch, best_m


# =============================================================================
# 8. FINAL SCALAR-INPUT VLM GATE
# =============================================================================


def configure_vlm_fold(
    fold_id,
    train_ids,
    val_ids,
    seg_train_dir,
    seg_val_dir,
    seg_test_dir,
    adaptive_train_dir,
    adaptive_val_dir,
    adaptive_test_dir,
    radii_m,
    lambda_ref,
):
    root = fold_root(fold_id) / "vlm"
    root.mkdir(parents=True, exist_ok=True)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (root / "test_predictions" / "school_probability_npy").mkdir(parents=True, exist_ok=True)
    (root / "test_predictions" / "school_binary_png").mkdir(parents=True, exist_ok=True)
    (root / "test_predictions" / "recovery_gate_npy").mkdir(parents=True, exist_ok=True)

    fold_dir = DATASET_ROOT / "folds" / "fold_{}".format(fold_id)

    VLM.DATASET_ROOT = DATASET_ROOT
    VLM.IMAGE_DIR = DATASET_ROOT / "images"
    VLM.MASK_DIR = DATASET_ROOT / "masks_png"
    VLM.SPLIT_DIR = fold_dir
    VLM.TRAIN_LIST = fold_dir / "train.txt"
    VLM.VAL_LIST = fold_dir / "val.txt"
    VLM.TEST_LIST = fold_dir / "test.txt"

    VLM.SEG_TRAIN_DIR = Path(seg_train_dir)
    VLM.SEG_VAL_DIR = Path(seg_val_dir)
    VLM.SEG_TEST_DIR = Path(seg_test_dir)
    VLM.ALL_SCHOOL_CACHE_DIR = ALL_SCHOOL_POI_CACHE

    VLM.ADAPTIVE_TRAIN_DIR = Path(adaptive_train_dir)
    VLM.ADAPTIVE_VAL_DIR = Path(adaptive_val_dir)
    VLM.ADAPTIVE_TEST_DIR = Path(adaptive_test_dir)

    VLM.SCENE_CACHE = UNIFIED_SCENE_CACHE
    VLM.SCENE_TRAIN = UNIFIED_SCENE_CACHE
    VLM.SCENE_VAL = UNIFIED_SCENE_CACHE
    VLM.SCENE_TEST = UNIFIED_SCENE_CACHE

    VLM.OUT_ROOT = root
    VLM.TRAIN_NPZ = root / "train_candidates.npz"
    VLM.VAL_NPZ = root / "val_candidates.npz"
    VLM.CKPT_DIR = root / "checkpoints"
    VLM.BEST_CKPT = root / "checkpoints" / "best_recovery_gate.pth"
    VLM.TRAIN_LOG_CSV = root / "training_log.csv"
    VLM.VAL_REFERENCE_CSV = root / "validation_reference.csv"

    VLM.TEST_OUT = root / "test_predictions"
    VLM.TEST_PROB_DIR = root / "test_predictions" / "school_probability_npy"
    VLM.TEST_BIN_DIR = root / "test_predictions" / "school_binary_png"
    VLM.TEST_GATE_DIR = root / "test_predictions" / "recovery_gate_npy"
    VLM.TEST_COMPARISON_CSV = root / "test_comparison.csv"
    VLM.TEST_DIAG_CSV = root / "test_recovery_diagnostic.csv"
    VLM.SUMMARY_JSON = root / "summary.json"

    VLM.EMPIRICAL_RADIUS_M = dict(radii_m)
    VLM.SELECTED_POI_LAMBDA = float(lambda_ref)

    # VLM.empirical_school_prior uses its own globals.
    return root


def train_scalar_vlm(fold_id, train_ids, val_ids, device):
    root = fold_root(fold_id) / "vlm"
    summary_path = root / "summary.json"

    if (
        SKIP_COMPLETED_STAGES
        and summary_path.exists()
        and VLM.BEST_CKPT.exists()
    ):
        s = read_json(summary_path)
        print("[Fold {}] scalar VLM already complete; skipping training.".format(fold_id))
        return int(s["best_epoch"]), s["best_val_metrics"]

    VLM.set_seed(SEED)
    train_data = VLM.build_train_candidates(train_ids)
    val_data = VLM.build_val_candidates(val_ids)
    best_epoch, _ = VLM.train_gate(train_data, val_data, device)

    ckpt = safe_torch_load(VLM.BEST_CKPT, map_location="cpu")
    best_val = dict(ckpt["val_metrics"])

    write_json(
        summary_path,
        {
            "fold_id": int(fold_id),
            "best_epoch": int(best_epoch),
            "best_val_metrics": best_val,
            "method": "scalar-input VLM scene-triggered recovery",
            "gate_inputs": [
                "P_seg",
                "S_poi",
                "P_base_adaptive_poi",
                "VLM_scene_margin",
            ],
            "candidate_rule": (
                "valid core AND P_seg<0.5 AND P_base<0.5 AND "
                "P_base>=0.20 AND VLM school/campus rank==1"
            ),
            "upstream_frozen": True,
        },
    )

    return int(best_epoch), best_val


def load_selected_vlm_gate(device):
    ckpt = safe_torch_load(VLM.BEST_CKPT, map_location="cpu")
    noop = bool(ckpt.get("noop", False))
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)

    if noop:
        model = None
    else:
        model = VLM.RecoveryGate().to(device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()

    return ckpt, noop, mean, std, model


@torch.no_grad()
def reconstruct_vlm_probability(gid, split_name, noop, mean, std, model, device):
    if split_name == "val":
        seg_root = VLM.SEG_VAL_DIR
        base_root = VLM.ADAPTIVE_VAL_DIR
    elif split_name == "test":
        seg_root = VLM.SEG_TEST_DIR
        base_root = VLM.ADAPTIVE_TEST_DIR
    else:
        raise ValueError(split_name)

    gt = VLM.read_mask(gid)
    p_seg = VLM.read_prob(seg_root, gid)
    s_poi = VLM.empirical_school_prior(gid)
    p_base = VLM.read_prob(base_root, gid)
    margin, rank = VLM.read_scene(UNIFIED_SCENE_CACHE, gid)

    feat = VLM.make_feature_map(p_seg, s_poi, p_base, margin)
    cand = VLM.candidate_mask(gt, p_seg, s_poi, p_base, rank)
    p_final = p_base.copy()

    if (not noop) and np.any(cand):
        raw = feat[cand].astype(np.float32)
        g = VLM.gate_values(model, raw, mean, std, device)
        base_c = p_base[cand]
        p_final[cand] = np.clip(
            base_c + g * (1.0 - base_c), 0.0, 1.0
        )

    if np.any(p_final + 1e-7 < p_base):
        raise RuntimeError("VLM monotonicity violated: {}".format(gid))

    return p_final.astype(np.float32)


@torch.no_grad()
def export_vlm_probabilities(ids, split_name, out_dir, noop, mean, std, model, device):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, gid in enumerate(ids, 1):
        p = reconstruct_vlm_probability(
            gid, split_name, noop, mean, std, model, device
        )
        save_prob(out_dir, gid, p)
        if i % 10 == 0 or i == len(ids):
            print("  VLM {} [{}/{}] {}".format(split_name.upper(), i, len(ids), gid))


# =============================================================================
# 9. FINAL AOI STAGE
# =============================================================================


def configure_aoi_fold(fold_id, vlm_val_dir, vlm_test_dir):
    root = fold_root(fold_id) / "aoi"
    root.mkdir(parents=True, exist_ok=True)

    AOI.REC = VLM
    AOI.DATASET_ROOT = DATASET_ROOT
    AOI.IMAGE_DIR = DATASET_ROOT / "images"
    AOI.MASK_DIR = DATASET_ROOT / "masks_png"
    AOI.SCHOOL_AOI_FILE = AOI_SHP

    AOI.OUT_ROOT = root
    AOI.VAL_VLM_PROB_DIR = Path(vlm_val_dir)
    AOI.VAL_SEARCH_CSV = root / "validation_search.csv"
    AOI.BEST_CONFIG_JSON = root / "best_config.json"
    AOI.SUMMARY_JSON = root / "summary.json"

    return root


def select_aoi_on_validation(fold_id, val_ids, vlm_val_dir):
    root = fold_root(fold_id) / "aoi"
    best_cfg = root / "best_config.json"

    if SKIP_COMPLETED_STAGES and best_cfg.exists():
        cfg = read_json(best_cfg)
        print("[Fold {}] AOI validation selection already complete; skipping search.".format(fold_id))
        return cfg

    aoi_cache = AOI.AOICache(AOI_SHP)
    val_cache = AOI.build_search_cache(
        val_ids,
        Path(vlm_val_dir),
        aoi_cache,
        "VALIDATION",
    )
    best = AOI.search_validation(val_cache)

    # search_validation writes AOI.BEST_CONFIG_JSON and returns best row in
    # the retained implementation. Read the saved config for stable API.
    cfg = read_json(AOI.BEST_CONFIG_JSON)
    return cfg



def apply_aoi_to_test(fold_id, test_ids, vlm_test_dir, out_dir, config):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aoi_cache = AOI.AOICache(AOI_SHP)

    sigma_m = float(config["sigma_m"])
    floor = float(config["floor"])

    removed_fp = 0
    removed_tp = 0
    active_aois = 0
    tiles_with_active = 0

    for i, gid in enumerate(test_ids, 1):
        image_path = AOI.find_image(gid)
        gt = VLM.read_mask(gid)
        p = read_prob(vlm_test_dir, gid)

        geo = AOI.build_active_aoi_geometry(
            image_path=image_path,
            prob=p,
            aoi_cache=aoi_cache,
        )

        corrected = AOI.correct_probability(
            p,
            geo["active_aoi_mask"],
            geo["outside_distance_m"],
            sigma_m,
            floor,
        )

        pred0 = (p >= THRESHOLD).astype(np.uint8)
        pred1 = (corrected >= THRESHOLD).astype(np.uint8)

        if np.any((pred0 == 0) & (pred1 == 1)):
            raise RuntimeError("AOI created positives: {}".format(gid))

        valid = gt != IGNORE_VALUE
        removed = valid & (pred0 == 1) & (pred1 == 0)
        removed_fp += int(np.sum(removed & (gt == 0)))
        removed_tp += int(np.sum(removed & (gt == 1)))

        active_aois += int(geo["n_active_aois"])
        if int(geo["n_active_aois"]) > 0:
            tiles_with_active += 1

        save_prob(out_dir, gid, corrected)

        if i % 10 == 0 or i == len(test_ids):
            print(
                "  AOI TEST [{}/{}] {} | active={} | cumulative removed FP={} TP={}"
                .format(
                    i,
                    len(test_ids),
                    gid,
                    geo["n_active_aois"],
                    removed_fp,
                    removed_tp,
                )
            )

    return {
        "removed_fp_pixels": int(removed_fp),
        "removed_tp_pixels": int(removed_tp),
        "active_aois": int(active_aois),
        "tiles_with_active_aoi": int(tiles_with_active),
        "test_tiles": int(len(test_ids)),
    }


# =============================================================================
# 10. REPORTING AFTER EVERY COMPLETED FOLD
# =============================================================================


def print_fold_results(fold_id, stage_results):
    print("\n" + "=" * 120)
    print("FOLD {} COMPLETE | PROCESS TEST METRICS".format(fold_id))
    print("=" * 120)

    for scope in ["Overall", "小学", "中学", "高等教育"]:
        print("\n[{}]".format(scope))
        print(
            "{:<34s} {:>9s} {:>9s} {:>9s} {:>9s} {:>9s}"
            .format("Stage", "P", "R", "F1", "IoU", "Acc")
        )
        for stage in STAGES:
            m = stage_results[stage][scope]
            print(
                "{:<34s} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f}"
                .format(
                    stage,
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["iou"],
                    m["accuracy"],
                )
            )

    print("\nOverall stage deltas vs pure SegFormer:")
    base = stage_results[STAGE_SEG]["Overall"]
    for stage in [STAGE_POI, STAGE_VLM, STAGE_FINAL]:
        m = stage_results[stage]["Overall"]
        print(
            "  {:<32s} dP={:+.4f} dR={:+.4f} dF1={:+.4f} dIoU={:+.4f}"
            .format(
                stage,
                m["precision"] - base["precision"],
                m["recall"] - base["recall"],
                m["f1"] - base["f1"],
                m["iou"] - base["iou"],
            )
        )


def save_fold_process_metrics(fold_id, stage_results):
    rows = []
    for stage in STAGES:
        rows.extend(stage_rows(fold_id, stage, stage_results[stage]))
    p = fold_root(fold_id) / "test_process_metrics.csv"
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")
    return rows


def update_cumulative_tables_and_print():
    all_rows = []
    for fold_id in range(5):
        p = PIPELINE_ROOT / "fold_{}".format(fold_id) / "test_process_metrics.csv"
        if p.exists():
            all_rows.extend(
                pd.read_csv(p, encoding="utf-8-sig").to_dict("records")
            )

    if not all_rows:
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(
        PIPELINE_ROOT / "all_folds_process_metrics_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Mean±SD among completed folds, updated immediately after every fold.
    srows = []
    for (stage, scope), sub in df.groupby(["stage", "scope"], sort=False):
        row = {
            "stage": stage,
            "scope": scope,
            "n_folds": int(sub["fold"].nunique()),
        }
        for metric_name in ["precision", "recall", "f1", "iou", "accuracy"]:
            x = sub[metric_name].astype(float).to_numpy()
            row[metric_name + "_mean"] = float(np.mean(x))
            row[metric_name + "_sd"] = (
                float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            )
        srows.append(row)

    sdf = pd.DataFrame(srows)
    sdf.to_csv(
        PIPELINE_ROOT / "completed_folds_process_mean_sd.csv",
        index=False,
        encoding="utf-8-sig",
    )

    completed = sorted(set(int(x) for x in df["fold"].tolist()))
    print("\nCompleted folds so far: {}".format(completed))
    print("Overall mean IoU over completed folds:")
    sub = sdf[sdf["scope"] == "Overall"]
    for stage in STAGES:
        x = sub[sub["stage"] == stage]
        if x.empty:
            continue
        r = x.iloc[0]
        print(
            "  {:<34s} {:.4f} ± {:.4f}"
            .format(stage, r["iou_mean"], r["iou_sd"])
        )


def finalize_fivefold_tables():
    p = PIPELINE_ROOT / "all_folds_process_metrics_long.csv"
    if not p.exists():
        return
    df = pd.read_csv(p, encoding="utf-8-sig")

    # Pooled micro metrics across all completed TEST folds.
    rows = []
    for (stage, scope), sub in df.groupby(["stage", "scope"], sort=False):
        c = [
            int(sub["TP"].sum()),
            int(sub["TN"].sum()),
            int(sub["FP"].sum()),
            int(sub["FN"].sum()),
        ]
        rows.append({"stage": stage, "scope": scope, **metrics(c)})

    pd.DataFrame(rows).to_csv(
        PIPELINE_ROOT / "fivefold_process_pooled_micro.csv",
        index=False,
        encoding="utf-8-sig",
    )


# =============================================================================
# 11. ONE COMPLETE FOLD
# =============================================================================


def run_one_fold(fold_id, device):
    fr = fold_root(fold_id)
    done_json = fr / "fold_complete.json"

    if SKIP_COMPLETED_FOLDS and done_json.exists():
        print("\n[Fold {}] already complete; reading saved results.".format(fold_id))
        p = fr / "test_process_metrics.csv"
        if p.exists():
            update_cumulative_tables_and_print()
        return

    print("\n" + "#" * 120)
    print("FINAL RETAINED PIPELINE | FOLD {}".format(fold_id))
    print("#" * 120)

    fold_paths = SEG.get_fold_paths(fold_id)
    train_ids = SEG.read_id_list(fold_paths["train"])
    val_ids = SEG.read_id_list(fold_paths["val"])
    test_ids = SEG.read_id_list(fold_paths["test"])

    # ------------------------------------------------------------------
    # 1) SegFormer pure visual baseline.
    # ------------------------------------------------------------------
    print("\n[1/4] SegFormer pure visual baseline")
    SEG.run_one_fold(fold_id, device)

    seg_root = SEG.OUT_BASE_ROOT / "fold_{}".format(fold_id)
    seg_val_dir = seg_root / "val_predictions_best" / "school_probability_npy"
    seg_test_dir = seg_root / "test_predictions" / "school_probability_npy"
    seg_train_dir = fr / "segformer_train_probability_npy"

    seg_model, seg_best_epoch = load_best_segformer(fold_id, device)
    export_seg_probabilities(
        seg_model, train_ids, seg_train_dir, device, "TRAIN"
    )

    # ------------------------------------------------------------------
    # POI radii + lambda_ref: TRAIN only.
    # ------------------------------------------------------------------
    radius_info = estimate_train_radii(fold_id, train_ids)
    radii_m = radius_info["radii_m"]

    print("\nFold {} empirical POI radii:".format(fold_id))
    print("  小学     = {:.0f} m".format(radii_m["primary"]))
    print("  中学     = {:.0f} m".format(radii_m["middle"]))
    print("  高等教育 = {:.0f} m".format(radii_m["higher"]))

    lambda_ref = search_lambda_train(
        fold_id,
        train_ids,
        seg_train_dir,
        radii_m,
    )

    # ------------------------------------------------------------------
    # 2) Final Adaptive visual-feature POI.
    # ------------------------------------------------------------------
    print("\n[2/4] Adaptive visual-feature POI")
    ap_paths, ap_best_epoch, ap_best_val = train_adaptive_poi(
        fold_id=fold_id,
        seg_model=seg_model,
        train_ids=train_ids,
        val_ids=val_ids,
        seg_train_dir=seg_train_dir,
        seg_val_dir=seg_val_dir,
        radii_m=radii_m,
        lambda_ref=lambda_ref,
        device=device,
    )

    # Configure final VLM module before training.
    configure_vlm_fold(
        fold_id=fold_id,
        train_ids=train_ids,
        val_ids=val_ids,
        seg_train_dir=seg_train_dir,
        seg_val_dir=seg_val_dir,
        seg_test_dir=seg_test_dir,
        adaptive_train_dir=ap_paths["train"],
        adaptive_val_dir=ap_paths["val"],
        adaptive_test_dir=ap_paths["test"],
        radii_m=radii_m,
        lambda_ref=lambda_ref,
    )

    # ------------------------------------------------------------------
    # 3) Final scalar-input VLM recovery Gate.
    # ------------------------------------------------------------------
    print("\n[3/4] Scalar-input VLM scene-triggered recovery")
    vlm_best_epoch, vlm_best_val = train_scalar_vlm(
        fold_id,
        train_ids,
        val_ids,
        device,
    )

    # Reconstruct frozen VLM validation probability for AOI VAL search.
    vlm_val_dir = fold_root(fold_id) / "vlm" / "val_probability_npy"
    ckpt, vlm_noop, vlm_mean, vlm_std, vlm_model = load_selected_vlm_gate(device)
    export_vlm_probabilities(
        val_ids,
        "val",
        vlm_val_dir,
        vlm_noop,
        vlm_mean,
        vlm_std,
        vlm_model,
        device,
    )

    # ------------------------------------------------------------------
    # 4) AOI parameters: VAL only.
    # ------------------------------------------------------------------
    print("\n[4/4] Active-AOI outside soft suppression")
    vlm_test_dir = fold_root(fold_id) / "vlm" / "test_probability_npy"
    configure_aoi_fold(fold_id, vlm_val_dir, vlm_test_dir)
    aoi_cfg = select_aoi_on_validation(fold_id, val_ids, vlm_val_dir)

    # ------------------------------------------------------------------
    # All learnable/tunable parameters are now frozen.
    # Process TEST once through the full chain.
    # ------------------------------------------------------------------
    print("\n" + "-" * 120)
    print("FOLD {} | FROZEN TEST PASS".format(fold_id))
    print("-" * 120)

    # Adaptive POI TEST.
    configure_adaptive_common(
        fold_id=fold_id,
        radii_m=radii_m,
        lambda_ref=lambda_ref,
        seg_train_dir=seg_train_dir,
    )

    if lambda_ref <= 1e-12:
        ap_paths["test"].mkdir(parents=True, exist_ok=True)
        for gid in test_ids:
            save_prob(ap_paths["test"], gid, read_prob(seg_test_dir, gid))
    else:
        ap_obj = safe_torch_load(ap_paths["ckpt"], map_location=device)
        ap_model = AP.FrozenSegVisualLambdaNet(
            segformer=seg_model,
            lambda_ref=lambda_ref,
        ).to(device)
        ap_model.gate.load_state_dict(ap_obj["gate_state_dict"], strict=True)
        ap_model.eval()
        export_adaptive_probabilities(
            ap_model,
            test_ids,
            seg_test_dir,
            radii_m,
            ap_paths["test"],
            device,
            "TEST",
        )
        del ap_model

    # VLM TEST.
    VLM.ADAPTIVE_TEST_DIR = ap_paths["test"]
    export_vlm_probabilities(
        test_ids,
        "test",
        vlm_test_dir,
        vlm_noop,
        vlm_mean,
        vlm_std,
        vlm_model,
        device,
    )

    if vlm_model is not None:
        del vlm_model

    # AOI TEST.
    final_test_dir = fold_root(fold_id) / "aoi" / "test_probability_npy"
    aoi_diag = apply_aoi_to_test(
        fold_id,
        test_ids,
        vlm_test_dir,
        final_test_dir,
        aoi_cfg,
    )

    # Process TEST metrics: all 4 final retained stages, all 4 scopes.
    stage_results = {
        STAGE_SEG: evaluate_probability_dir(test_ids, seg_test_dir),
        STAGE_POI: evaluate_probability_dir(test_ids, ap_paths["test"]),
        STAGE_VLM: evaluate_probability_dir(test_ids, vlm_test_dir),
        STAGE_FINAL: evaluate_probability_dir(test_ids, final_test_dir),
    }

    save_fold_process_metrics(fold_id, stage_results)

    params = {
        "fold": int(fold_id),
        "pipeline_mode": PIPELINE_MODE,
        "segformer_best_epoch": int(seg_best_epoch),
        "radius_primary_m": float(radii_m["primary"]),
        "radius_middle_m": float(radii_m["middle"]),
        "radius_higher_m": float(radii_m["higher"]),
        "radius_primary_raw_p90_m": float(radius_info["raw_p90_m"]["primary"]),
        "radius_middle_raw_p90_m": float(radius_info["raw_p90_m"]["middle"]),
        "radius_higher_raw_p90_m": float(radius_info["raw_p90_m"]["higher"]),
        "radius_n_train_buildings": radius_info["n_train_buildings"],
        "radius_source": radius_info["source"],
        "lambda_ref": float(lambda_ref),
        "adaptive_poi_best_epoch": int(ap_best_epoch),
        "vlm_best_epoch": int(vlm_best_epoch),
        "aoi_sigma_m": float(aoi_cfg["sigma_m"]),
        "aoi_floor": float(aoi_cfg["floor"]),
        "aoi_test_diagnostic": aoi_diag,
    }
    write_json(fr / "fold_parameters.json", params)

    print_fold_results(fold_id, stage_results)

    write_json(
        done_json,
        {
            "fold": int(fold_id),
            "status": "complete",
            "parameters": params,
            "process_metrics_file": str(fr / "test_process_metrics.csv"),
        },
    )

    # Cumulative table is updated and printed immediately after this fold.
    update_cumulative_tables_and_print()

    del seg_model
    if device.type == "cuda":
        torch.cuda.empty_cache()


# =============================================================================
# 12. INPUT CHECKS / MAIN
# =============================================================================


def check_static_inputs():
    required = [
        DATASET_ROOT,
        BUILDING_SHP,
        CORE_SHP,
        AOI_SHP,
        CORE_MATCH_AUDIT,
        BUILDING_R95_CSV,
        ALL_SCHOOL_POI_CACHE,
    ]
    for p in required:
        if not Path(p).exists():
            raise FileNotFoundError("Required input missing:\n{}".format(p))

    # Point all retained modules at the same prepared dataset / baseline root.
    SEG.DATASET_ROOT = DATASET_ROOT
    SEG.IMAGE_DIR = DATASET_ROOT / "images"
    SEG.MASK_DIR = DATASET_ROOT / "masks_png"
    SEG.TYPE_MASK_DIR = DATASET_ROOT / "masks_type_png"
    SEG.FOLDS_DIR = DATASET_ROOT / "folds"
    SEG.VERSION_FILE = DATASET_ROOT / "dataset_version.txt"
    SEG.DATASET_VERSION = SEG.VERSION_FILE.read_text(encoding="utf-8").strip()

    AP.DATASET_ROOT = DATASET_ROOT
    AP.IMAGE_DIR = DATASET_ROOT / "images"
    AP.MASK_DIR = DATASET_ROOT / "masks_png"
    AP.TYPE_MASK_DIR = DATASET_ROOT / "masks_type_png"
    AP.ALL_SCHOOL_CACHE_DIR = ALL_SCHOOL_POI_CACHE

    VLM.DATASET_ROOT = DATASET_ROOT
    VLM.IMAGE_DIR = DATASET_ROOT / "images"
    VLM.MASK_DIR = DATASET_ROOT / "masks_png"
    VLM.ALL_SCHOOL_CACHE_DIR = ALL_SCHOOL_POI_CACHE

    AOI.REC = VLM
    AOI.DATASET_ROOT = DATASET_ROOT
    AOI.IMAGE_DIR = DATASET_ROOT / "images"
    AOI.MASK_DIR = DATASET_ROOT / "masks_png"


def main():
    PIPELINE_ROOT.mkdir(parents=True, exist_ok=True)
    check_static_inputs()
    prepare_unified_scene_cache()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda:0")

    print("=" * 120)
    print("FINAL RETAINED SCHOOL MULTISOURCE PIPELINE | STAGE-WISE 5-FOLD")
    print("=" * 120)
    print("Mode:", PIPELINE_MODE)
    print("RUN_FOLDS:", RUN_FOLDS)
    print("GPU:", torch.cuda.get_device_name(0))
    print()
    print("Retained route ONLY:")
    print("  SegFormer")
    print("    -> empirical-radius POI + SegFormer visual-feature adaptive lambda(x)")
    print("    -> scalar-input VLM scene-triggered recovery")
    print("    -> active-AOI outside soft suppression")
    print()
    print("This is stage-wise training. Upstream stages are frozen downstream.")
    print("After EVERY completed fold, all process metrics are printed immediately.")

    for fold_id in RUN_FOLDS:
        run_one_fold(int(fold_id), device)

    finalize_fivefold_tables()

    print("\n" + "=" * 120)
    print("REQUESTED FOLDS FINISHED")
    print("=" * 120)
    print("Main outputs:")
    print(" ", PIPELINE_ROOT / "all_folds_process_metrics_long.csv")
    print(" ", PIPELINE_ROOT / "completed_folds_process_mean_sd.csv")
    print(" ", PIPELINE_ROOT / "fivefold_process_pooled_micro.csv")
    print(" Per-fold parameters/results are under fold_0 ... fold_4.")


if __name__ == "__main__":
    main()
