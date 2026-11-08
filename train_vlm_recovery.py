# -*- coding: utf-8 -*-
r"""
Fold 0: adaptive visual-feature POI + VLM scene-triggered recovery
=================================================================

This is the VLM continuation of the CURRENT adaptive-POI experiment.

Important architecture relation
-------------------------------
The adaptive visual-feature POI branch REPLACES the previous fixed global-lambda
POI branch. It is not appended after fixed POI.

Current baseline for this script:

    SegFormer
        -> empirical-radius POI
        -> SegFormer visual-feature adaptive lambda(x)
        = Adaptive visual-feature POI

Then VLM is added on top:

    Adaptive visual-feature POI
        -> VLM scene-triggered recovery

The VLM role is unchanged from the latest 55b design:
    - VLM only RECOVERS additional positives;
    - it never suppresses the adaptive-POI baseline.

Candidate pixel
---------------
A pixel is eligible only when all conditions hold:

    1) valid evaluation core
    2) P_seg < 0.5
    3) P_adaptive < 0.5
    4) P_adaptive >= VISUAL_FLOOR (=0.20)
    5) DGTRS school/campus semantic group rank == 1

Gate input
----------
For each candidate pixel:

    [P_seg, S_poi, P_adaptive, VLM_scene_margin]

where:
    P_seg       = pure SegFormer school probability
    S_poi       = fixed empirical-radius POI spatial prior
    P_adaptive  = selected visual-feature adaptive POI probability
    VLM margin  = DGTRS/LRSCLIP school/campus scene-semantic margin

Recovery
--------
    P_final = P_adaptive + g(x) * (1 - P_adaptive)

Therefore:
    P_final >= P_adaptive

Training / selection
--------------------
- adaptive-POI probabilities are frozen;
- only the tiny 4-input recovery MLP is trained;
- TRAIN candidate pixels only;
- best epoch selected by FULL validation-core micro IoU;
- epoch 0 / no-op is allowed;
- held-out TEST evaluated once;
- threshold fixed at 0.5.

Required local files beside this script
---------------------------------------
    export_adaptive_poi_train_val_fold0.py  (run first in larse)
    poi_visualfeature_gate_fold0.py          (needed by exporter only)

Required adaptive-POI probability cache
---------------------------------------
Run first:
    conda activate larse
    python export_adaptive_poi_train_val_fold0.py

Then run this VLM script:
    conda activate school_vlm
    python vlm_after_adaptive_poi_fold0.py

Scene cache
-----------
The DGTRS scene maps depend only on the image/text configuration, not on the POI
baseline. Therefore this script reuses the already-generated scene cache from:

    E:\larse_result\vlm_scene_recovery_empirical_poi_fold0\scene_cache

If any map is missing, the same DGTRS code can generate it.

No rasterio is used.
"""

from __future__ import print_function

import csv
import json
import math
import random
import sys
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# 1. PATHS
# =============================================================================

DATASET_ROOT = Path(
    r"G:\SH\上海市\学校实验区_500m_B150m\0924\segformer_binary_5fold_dataset"
)

IMAGE_DIR = DATASET_ROOT / "images"
MASK_DIR = DATASET_ROOT / "masks_png"

SPLIT_DIR = DATASET_ROOT / "folds" / "fold_0"
TRAIN_LIST = SPLIT_DIR / "train.txt"
VAL_LIST = SPLIT_DIR / "val.txt"
TEST_LIST = SPLIT_DIR / "test.txt"

SEG_ROOT = Path(
    r"E:\larse_result\segformer_b2_binary_5fold\fold_0"
)

# Pure SegFormer TRAIN probabilities were already exported by the previous
# formal POI experiment. VAL / TEST come from the Fold-0 SegFormer run.
SEG_TRAIN_DIR = Path(
    r"E:\larse_result\segformer_poi_competitive_5fold\fold_0"
    r"\train_visual_probabilities\school_probability_npy"
)

SEG_VAL_DIR = (
    SEG_ROOT
    / "val_predictions_best"
    / "school_probability_npy"
)

SEG_TEST_DIR = (
    SEG_ROOT
    / "test_predictions"
    / "school_probability_npy"
)

# All raw school POIs, already cached as three Gaussian base-prior channels.
ALL_SCHOOL_CACHE_DIR = Path(
    r"E:\larse_result\poi_all_school_base_prior_cache\cache"
)

# Exact configuration of the immediately preceding empirical-radius POI run.
POI_STAGE_ROOT = Path(
    r"E:\larse_result\segformer_poi_empirical_radius_logit_fold0"
)
POI_CONFIG_JSON = POI_STAGE_ROOT / "selected_config.json"
POI_VAL_REFERENCE_JSON = POI_STAGE_ROOT / "validation_summary.json"

# Selected adaptive visual-feature POI stage.
ADAPTIVE_POI_ROOT = Path(
    r"E:\larse_result\segformer_poi_visualfeature_gate_fold0"
)

ADAPTIVE_TRAIN_DIR = (
    ADAPTIVE_POI_ROOT
    / "adaptive_probability_cache"
    / "train"
)

ADAPTIVE_VAL_DIR = (
    ADAPTIVE_POI_ROOT
    / "adaptive_probability_cache"
    / "val"
)

ADAPTIVE_TEST_DIR = (
    ADAPTIVE_POI_ROOT
    / "test_predictions"
    / "adaptive_probability_npy"
)

ADAPTIVE_CACHE_SUMMARY_JSON = (
    ADAPTIVE_POI_ROOT
    / "adaptive_probability_cache_summary.json"
)

ADAPTIVE_TEST_SUMMARY_JSON = (
    ADAPTIVE_POI_ROOT
    / "test_summary.json"
)

LARSE_LABEL_FILE = (
    Path(__file__).resolve().parent / "configs" / "scene_classes.txt"
)

DGTRS_ROOT = (
    Path(__file__).resolve().parent / "third_party" / "DGTRS"
)


DGTRS_CKPT = (
    Path(__file__).resolve().parent
    / "pretrained"
    / "dgtrs"
    / "DGTRS-CLIP-ViT-B-16.pt"
)

OUT_ROOT = Path(
    r"E:\larse_result\vlm_scene_recovery_adaptive_poi_fold0"
)

# Use a fresh scene cache for the current 421-tile dataset. Scene maps depend
# only on the image and text configuration, but fresh generation avoids any
# ambiguity about old dataset versions.
SCENE_CACHE = Path(
    r"E:\larse_result\vlm_scene_recovery_empirical_poi_fold0\scene_cache"
)
SCENE_TRAIN = SCENE_CACHE / "train"
SCENE_VAL = SCENE_CACHE / "val"
SCENE_TEST = SCENE_CACHE / "test"

# Disable external test-scene reuse: missing maps are generated from current
# IMAGE_DIR instead.
EXP54_ROOT = OUT_ROOT / "_no_external_scene_reuse"
EXP54_MARGIN_DIR = EXP54_ROOT / "vlm_margin_heatmap_npy"
EXP54_RANK_DIR = EXP54_ROOT / "vlm_group_rank_npy"

TRAIN_NPZ = OUT_ROOT / "train_candidates.npz"
VAL_NPZ = OUT_ROOT / "val_candidates.npz"

CKPT_DIR = OUT_ROOT / "checkpoints"
BEST_CKPT = CKPT_DIR / "best_recovery_gate.pth"

TRAIN_LOG_CSV = OUT_ROOT / "training_log.csv"
VAL_REFERENCE_CSV = OUT_ROOT / "validation_reference.csv"

TEST_OUT = OUT_ROOT / "test_predictions"
TEST_PROB_DIR = TEST_OUT / "school_probability_npy"
TEST_BIN_DIR = TEST_OUT / "school_binary_png"
TEST_GATE_DIR = TEST_OUT / "recovery_gate_npy"

TEST_COMPARISON_CSV = OUT_ROOT / "test_comparison.csv"
TEST_DIAG_CSV = OUT_ROOT / "test_recovery_diagnostic.csv"
SUMMARY_JSON = OUT_ROOT / "summary.json"


# =============================================================================
# 2. FIXED SETTINGS
# =============================================================================

SEED = 42

THRESHOLD = 0.50

# Quick diagnostic design choice:
# only weak-but-nontrivial visual evidence is eligible for VLM recovery.
VISUAL_FLOOR = 0.20

PATCH_SIZE = 224
STRIDE = 112
VLM_BATCH_SIZE = 32

IGNORE_VALUE = 255

GATE_H1 = 16
GATE_H2 = 8

EPOCHS = 20
BATCH_SIZE = 8192
LR = 1e-3
WEIGHT_DECAY = 1e-4

# TRAIN sampling only.
MAX_POS_PER_TILE = 2500
NEG_TO_POS_RATIO = 2.0
NEG_IF_NO_POS = 1500
MAX_TRAIN_SAMPLES = 1_500_000

EPS = 1e-7


# Filled by load_poi_stage_config() in main().
EMPIRICAL_RADIUS_M = {
    "primary": None,
    "middle": None,
    "higher": None,
}
SELECTED_POI_LAMBDA = None
POI_PROB_EPS = 1e-6

TYPE_TO_CACHE_INDEX = {
    "primary": 0,
    "middle": 1,
    "higher": 2,
}

EXTRA_POSITIVE_PROMPTS = [
    "school",
    "school campus",
    "campus",
    "educational institution",
    "primary school",
    "middle school",
    "high school",
    "university campus",
    "college campus",
    "educational buildings",
    "school playground",
    "school sports field",
    "campus playground",
]

IMAGE_SUFFIXES = [
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
]


# =============================================================================
# 3. GENERAL IO
# =============================================================================

def ensure_dirs():
    for d in [
        OUT_ROOT,
        SCENE_CACHE,
        SCENE_TRAIN,
        SCENE_VAL,
        SCENE_TEST,
        CKPT_DIR,
        TEST_OUT,
        TEST_PROB_DIR,
        TEST_BIN_DIR,
        TEST_GATE_DIR,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True
        )


def check_inputs():
    required = [
        DATASET_ROOT,
        IMAGE_DIR,
        MASK_DIR,
        TRAIN_LIST,
        VAL_LIST,
        TEST_LIST,
        SEG_TRAIN_DIR,
        SEG_VAL_DIR,
        SEG_TEST_DIR,
        ALL_SCHOOL_CACHE_DIR,
        POI_CONFIG_JSON,
        ADAPTIVE_TRAIN_DIR,
        ADAPTIVE_VAL_DIR,
        ADAPTIVE_TEST_DIR,
        ADAPTIVE_CACHE_SUMMARY_JSON,
        ADAPTIVE_TEST_SUMMARY_JSON,
        LARSE_LABEL_FILE,
        DGTRS_ROOT,
        DGTRS_CKPT,
    ]

    for p in required:
        if not Path(p).exists():
            raise FileNotFoundError(
                "Required input not found:\n{}"
                .format(p)
            )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_ids(path):
    ids = [
        x.strip()
        for x in Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate IDs in:\n{}"
            .format(path)
        )

    return ids


def find_file(
    root,
    gid,
    suffixes,
):
    root = Path(root)

    for s in suffixes:
        p = root / (gid + s)

        if p.exists():
            return p

    matches = []

    for s in suffixes:
        matches.extend(
            root.glob(
                gid
                + "*"
                + s
            )
        )

    matches = [
        p
        for p in matches
        if p.is_file()
    ]

    if len(matches) == 1:
        return matches[0]

    if not matches:
        raise FileNotFoundError(
            "Cannot find {} under:\n{}"
            .format(
                gid,
                root
            )
        )

    raise RuntimeError(
        "Ambiguous files for {} under:\n{}"
        .format(
            gid,
            root
        )
    )


def read_rgb(gid):
    p = find_file(
        IMAGE_DIR,
        gid,
        IMAGE_SUFFIXES,
    )

    with Image.open(
        str(p)
    ) as im:
        return im.convert(
            "RGB"
        ).copy()


def read_mask(gid):
    """
    Current dataset masks are already authoritative complete-core targets:
        0   background inside the true core
        1   school building inside the true core
        255 outside the true core (ignore/context)
    """
    p = find_file(
        MASK_DIR,
        gid,
        [".png"],
    )

    with Image.open(str(p)) as im:
        arr = np.asarray(im, dtype=np.uint8)

    values = set(np.unique(arr).tolist())
    if not values.issubset({0, 1, 255}):
        raise RuntimeError(
            "Unexpected mask values for {}: {}"
            .format(gid, sorted(values))
        )

    return arr


def prob_path(
    root,
    gid
):
    root = Path(root)

    candidates = [
        root / (
            gid
            + "_school_probability.npy"
        ),
        root / (
            gid
            + ".npy"
        ),
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        "Probability missing for {} under:\n{}"
        .format(
            gid,
            root
        )
    )


def read_prob(
    root,
    gid
):
    p = prob_path(
        root,
        gid
    )

    arr = np.load(
        str(p)
    ).astype(
        np.float32
    )

    if arr.ndim != 2:
        raise RuntimeError(
            "Probability must be HxW:\n{}"
            .format(p)
        )

    if not np.isfinite(
        arr
    ).all():
        raise RuntimeError(
            "NaN/Inf in probability:\n{}"
            .format(p)
        )

    return np.clip(
        arr,
        0.0,
        1.0
    )


def write_csv(
    path,
    rows
):
    if not rows:
        return

    with open(
        str(path),
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            )
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


# =============================================================================
# 4. TEXT / DGTRS
# =============================================================================

def unique_preserve_order(items):
    seen = set()
    out = []

    for x in items:
        x = str(x).strip()

        if not x:
            continue

        key = x.lower()

        if key not in seen:
            out.append(x)
            seen.add(key)

    return out


def build_text_config():
    lines = LARSE_LABEL_FILE.read_text(
        encoding="utf-8"
    ).splitlines()

    original = [
        line.strip()
        .split(",")[-1]
        .split(";")[0]
        .strip()
        for line in lines
    ][1:]

    if len(original) != 12:
        raise RuntimeError(
            "Expected 12 original LaRSE classes, got {}."
            .format(
                len(original)
            )
        )

    school_matches = [
        i
        for i, name in enumerate(
            [
                x.lower()
                for x in original
            ]
        )
        if (
            name == "school"
            or name == "education"
            or "school" in name
            or "education" in name
        )
    ]

    if len(school_matches) != 1:
        raise RuntimeError(
            "Cannot uniquely identify original school/education class."
        )

    school_idx = int(
        school_matches[0]
    )

    positive_prompts = unique_preserve_order(
        [
            original[
                school_idx
            ]
        ]
        + EXTRA_POSITIVE_PROMPTS
    )

    negative_labels = [
        x
        for i, x in enumerate(
            original
        )
        if i != school_idx
    ]

    return {
        "positive_prompts": positive_prompts,
        "negative_labels": negative_labels,
        "all_texts": (
            positive_prompts
            + negative_labels
        ),
        "n_positive": int(
            len(
                positive_prompts
            )
        ),
    }


def load_dgtrs(
    text_cfg
):
    if str(
        DGTRS_ROOT
    ) not in sys.path:
        sys.path.insert(
            0,
            str(
                DGTRS_ROOT
            )
        )

    from model import longclip

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        "DGTRS device:",
        device
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            )
        )

    model, preprocess = longclip.load(
        str(
            DGTRS_CKPT
        ),
        device=str(
            device
        )
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    tokens = longclip.tokenize(
        text_cfg[
            "all_texts"
        ]
    ).to(
        device
    )

    with torch.no_grad():
        text_features = (
            model.encode_text(
                tokens
            )
            .float()
        )

        text_features = (
            text_features
            / text_features.norm(
                dim=-1,
                keepdim=True
            ).clamp_min(
                1e-6
            )
        )

    return (
        model,
        preprocess,
        text_features,
        device,
    )


# =============================================================================
# 5. PATCH SCENE MAPS
# =============================================================================

def sliding_starts(
    length,
    size,
    stride
):
    if length <= size:
        return [0]

    starts = list(
        range(
            0,
            length
            - size
            + 1,
            stride
        )
    )

    last = (
        length
        - size
    )

    if starts[-1] != last:
        starts.append(
            last
        )

    return starts


def pad_if_small(rgb):
    w, h = rgb.size

    out_w = max(
        w,
        PATCH_SIZE
    )

    out_h = max(
        h,
        PATCH_SIZE
    )

    if (
        out_w == w
        and out_h == h
    ):
        return (
            rgb,
            (
                h,
                w,
            )
        )

    canvas = Image.new(
        "RGB",
        (
            out_w,
            out_h,
        ),
        (
            0,
            0,
            0,
        )
    )

    canvas.paste(
        rgb,
        (
            0,
            0,
        )
    )

    return (
        canvas,
        (
            h,
            w,
        )
    )


def scene_paths(
    cache_dir,
    gid
):
    return (
        Path(cache_dir)
        / (
            gid
            + "_vlm_margin.npy"
        ),
        Path(cache_dir)
        / (
            gid
            + "_vlm_group_rank.npy"
        ),
    )


@torch.no_grad()
def infer_scene_map(
    gid,
    rgb,
    model,
    preprocess,
    text_features,
    device,
    text_cfg,
):
    rgb_pad, original_hw = pad_if_small(
        rgb
    )

    orig_h, orig_w = original_hw

    width, height = rgb_pad.size

    xs = sliding_starts(
        width,
        PATCH_SIZE,
        STRIDE,
    )

    ys = sliding_starts(
        height,
        PATCH_SIZE,
        STRIDE,
    )

    patches = []
    boxes = []

    for top in ys:
        for left in xs:
            patches.append(
                rgb_pad.crop(
                    (
                        left,
                        top,
                        left + PATCH_SIZE,
                        top + PATCH_SIZE,
                    )
                )
            )

            boxes.append(
                (
                    left,
                    top,
                    left + PATCH_SIZE,
                    top + PATCH_SIZE,
                )
            )

    sims_all = []

    for start in range(
        0,
        len(
            patches
        ),
        VLM_BATCH_SIZE,
    ):
        sub = patches[
            start:
            start
            + VLM_BATCH_SIZE
        ]

        x = torch.stack(
            [
                preprocess(
                    patch
                )
                for patch in sub
            ],
            dim=0,
        ).to(
            device
        )

        image_features = (
            model.encode_image(
                x
            )
            .float()
        )

        image_features = (
            image_features
            / image_features.norm(
                dim=-1,
                keepdim=True
            ).clamp_min(
                1e-6
            )
        )

        sims = (
            image_features
            @ text_features.T
        )

        sims_all.append(
            sims.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

    sims = np.concatenate(
        sims_all,
        axis=0
    )

    n_pos = int(
        text_cfg[
            "n_positive"
        ]
    )

    pos = sims[
        :,
        :n_pos
    ]

    neg = sims[
        :,
        n_pos:
    ]

    best_pos = np.max(
        pos,
        axis=1
    )

    best_neg = np.max(
        neg,
        axis=1
    )

    margin = (
        best_pos
        - best_neg
    ).astype(
        np.float32
    )

    rank = (
        1
        + np.sum(
            neg
            > best_pos[
                :,
                None
            ],
            axis=1
        )
    ).astype(
        np.uint8
    )

    margin_map = np.full(
        (
            height,
            width
        ),
        -np.inf,
        dtype=np.float32
    )

    rank_map = np.full(
        (
            height,
            width
        ),
        255,
        dtype=np.uint8
    )

    for i, (
        left,
        top,
        right,
        bottom,
    ) in enumerate(
        boxes
    ):
        margin_map[
            top:bottom,
            left:right
        ] = np.maximum(
            margin_map[
                top:bottom,
                left:right
            ],
            margin[
                i
            ]
        )

        rank_map[
            top:bottom,
            left:right
        ] = np.minimum(
            rank_map[
                top:bottom,
                left:right
            ],
            rank[
                i
            ]
        )

    margin_map = margin_map[
        :orig_h,
        :orig_w
    ]

    rank_map = rank_map[
        :orig_h,
        :orig_w
    ]

    if np.any(
        ~np.isfinite(
            margin_map
        )
    ):
        raise RuntimeError(
            "Uncovered scene pixel for {}."
            .format(gid)
        )

    if np.any(
        rank_map
        == 255
    ):
        raise RuntimeError(
            "Uncovered scene rank for {}."
            .format(gid)
        )

    return (
        margin_map,
        rank_map,
    )


def ensure_scene_maps(
    ids,
    split_name,
    cache_dir,
    model,
    preprocess,
    text_features,
    device,
    text_cfg,
):
    cache_dir = Path(
        cache_dir
    )

    cache_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    missing = []

    for gid in ids:
        mp, rp = scene_paths(
            cache_dir,
            gid
        )

        if not (
            mp.exists()
            and rp.exists()
        ):
            # Reuse experiment-54 TEST cache if possible.
            if (
                split_name == "test"
            ):
                src_m = (
                    EXP54_MARGIN_DIR
                    / (
                        gid
                        + "_vlm_margin.npy"
                    )
                )

                src_r = (
                    EXP54_RANK_DIR
                    / (
                        gid
                        + "_vlm_group_rank.npy"
                    )
                )

                if (
                    src_m.exists()
                    and src_r.exists()
                ):
                    m = np.load(
                        str(
                            src_m
                        )
                    )

                    r = np.load(
                        str(
                            src_r
                        )
                    )

                    np.save(
                        str(
                            mp
                        ),
                        m.astype(
                            np.float32
                        )
                    )

                    np.save(
                        str(
                            rp
                        ),
                        r.astype(
                            np.uint8
                        )
                    )

                    continue

            missing.append(
                gid
            )

    print()
    print(
        "Scene {} cache: {} existing/reused, {} missing."
        .format(
            split_name.upper(),
            len(
                ids
            )
            - len(
                missing
            ),
            len(
                missing
            ),
        )
    )

    for i, gid in enumerate(
        missing,
        1
    ):
        rgb = read_rgb(
            gid
        )

        margin, rank = infer_scene_map(
            gid,
            rgb,
            model,
            preprocess,
            text_features,
            device,
            text_cfg,
        )

        mp, rp = scene_paths(
            cache_dir,
            gid
        )

        np.save(
            str(
                mp
            ),
            margin.astype(
                np.float32
            )
        )

        np.save(
            str(
                rp
            ),
            rank.astype(
                np.uint8
            )
        )

        if (
            i % 10 == 0
            or i == len(
                missing
            )
        ):
            print(
                "  scene {} [{}/{}] {}"
                .format(
                    split_name,
                    i,
                    len(
                        missing
                    ),
                    gid,
                )
            )


def read_scene(
    cache_dir,
    gid
):
    mp, rp = scene_paths(
        cache_dir,
        gid
    )

    margin = np.load(
        str(
            mp
        )
    ).astype(
        np.float32
    )

    rank = np.load(
        str(
            rp
        )
    ).astype(
        np.uint8
    )

    return (
        margin,
        rank,
    )


# =============================================================================
# 6A. CURRENT EMPIRICAL-RADIUS POI BASELINE
# =============================================================================

def load_poi_stage_config():
    global EMPIRICAL_RADIUS_M
    global SELECTED_POI_LAMBDA

    cfg = json.loads(
        POI_CONFIG_JSON.read_text(encoding="utf-8")
    )

    EMPIRICAL_RADIUS_M = {
        "primary": float(cfg["radius_primary_m"]),
        "middle": float(cfg["radius_middle_m"]),
        "higher": float(cfg["radius_higher_m"]),
    }

    SELECTED_POI_LAMBDA = float(
        cfg["selected_lambda"]
    )

    if SELECTED_POI_LAMBDA < 0.0:
        raise RuntimeError("POI lambda must be non-negative.")

    print()
    print("Loaded empirical-radius POI stage:")
    print(
        "  primary={:.0f} m, middle={:.0f} m, higher={:.0f} m, lambda={:.2f}"
        .format(
            EMPIRICAL_RADIUS_M["primary"],
            EMPIRICAL_RADIUS_M["middle"],
            EMPIRICAL_RADIUS_M["higher"],
            SELECTED_POI_LAMBDA,
        )
    )


def load_school_base(gid):
    p = ALL_SCHOOL_CACHE_DIR / (str(gid) + ".npz")
    if not p.exists():
        raise FileNotFoundError(p)

    with np.load(str(p), allow_pickle=False) as z:
        base = z["school_base"].astype(np.float32)
        base_sigma = float(z["base_sigma_m"][0])

    if base.ndim != 3 or base.shape[0] != 3:
        raise RuntimeError(
            "{}: school_base shape={}, expected [3,H,W]"
            .format(gid, base.shape)
        )

    return base, base_sigma


def base_gaussian_to_distance(base_prior, base_sigma_m):
    base = np.clip(
        base_prior.astype(np.float64, copy=False),
        0.0,
        1.0,
    )

    distance = np.full(
        base.shape,
        np.inf,
        dtype=np.float32,
    )

    positive = base > 0.0
    if np.any(positive):
        x = np.maximum(base[positive], 1e-300)
        d = float(base_sigma_m) * np.sqrt(
            np.maximum(0.0, -2.0 * np.log(x))
        )
        distance[positive] = d.astype(np.float32)

    return distance


def cosine_prior_from_distance(distance_m, radius_m):
    radius_m = float(radius_m)
    out = np.zeros(distance_m.shape, dtype=np.float32)

    inside = distance_m <= radius_m
    if np.any(inside):
        x = distance_m[inside].astype(np.float64) / radius_m
        out[inside] = (
            0.5 * (1.0 + np.cos(np.pi * x))
        ).astype(np.float32)

    return out


def empirical_school_prior(gid):
    base, base_sigma = load_school_base(gid)
    priors = []

    for name in ["primary", "middle", "higher"]:
        idx = TYPE_TO_CACHE_INDEX[name]
        d = base_gaussian_to_distance(
            base[idx],
            base_sigma,
        )
        priors.append(
            cosine_prior_from_distance(
                d,
                EMPIRICAL_RADIUS_M[name],
            )
        )

    return np.maximum.reduce(priors).astype(np.float32)


def np_logit(p):
    p = np.clip(
        p.astype(np.float64, copy=False),
        POI_PROB_EPS,
        1.0 - POI_PROB_EPS,
    )
    return np.log(p / (1.0 - p))


def np_sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out.astype(np.float32)


def current_poi_base(p_seg, s_poi):
    if SELECTED_POI_LAMBDA is None:
        raise RuntimeError("POI stage configuration has not been loaded.")

    p_base = np_sigmoid(
        np_logit(p_seg)
        + float(SELECTED_POI_LAMBDA)
        * s_poi.astype(np.float64)
    )

    # The preceding POI stage is positive-only.
    if np.any(p_base + 1e-7 < p_seg):
        raise RuntimeError(
            "Empirical-radius POI reconstruction violated monotonicity."
        )

    return p_base


# =============================================================================
# 6. METRICS
# =============================================================================

def counts(
    gt,
    pred
):
    valid = (
        gt
        != IGNORE_VALUE
    )

    g = (
        gt[
            valid
        ]
        == 1
    )

    p = (
        pred[
            valid
        ]
        == 1
    )

    tp = int(
        np.sum(
            g
            & p
        )
    )

    tn = int(
        np.sum(
            (~g)
            & (~p)
        )
    )

    fp = int(
        np.sum(
            (~g)
            & p
        )
    )

    fn = int(
        np.sum(
            g
            & (~p)
        )
    )

    return (
        tp,
        tn,
        fp,
        fn
    )


def add_counts(
    total,
    c
):
    for i in range(
        4
    ):
        total[
            i
        ] += int(
            c[
                i
            ]
        )


def metrics_from_counts(
    tp,
    tn,
    fp,
    fn
):
    precision = (
        tp
        / max(
            1,
            tp
            + fp
        )
    )

    recall = (
        tp
        / max(
            1,
            tp
            + fn
        )
    )

    f1 = (
        2.0
        * tp
        / max(
            EPS,
            2.0
            * tp
            + fp
            + fn
        )
    )

    iou = (
        tp
        / max(
            1,
            tp
            + fp
            + fn
        )
    )

    accuracy = (
        tp
        + tn
    ) / max(
        1,
        tp
        + tn
        + fp
        + fn
    )

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "precision": float(
            precision
        ),
        "recall": float(
            recall
        ),
        "f1": float(
            f1
        ),
        "iou": float(
            iou
        ),
        "accuracy": float(
            accuracy
        ),
    }


# =============================================================================
# 7. CANDIDATE FEATURES
# =============================================================================

def make_feature_map(
    p_seg,
    s_poi,
    p_base,
    margin,
):
    """
    VLM feature map for the CURRENT adaptive-POI baseline.

    p_base is read from the already-selected adaptive visual-feature POI stage.
    It is NOT reconstructed by the old fixed global-lambda formula.
    """
    if not (
        p_seg.shape
        == s_poi.shape
        == p_base.shape
        == margin.shape
    ):
        raise RuntimeError(
            "Feature-map shape mismatch."
        )

    # Adaptive POI remains positive-only relative to pure SegFormer.
    if np.any(
        p_base
        + 1e-7
        < p_seg
    ):
        raise RuntimeError(
            "Adaptive POI baseline fell below pure SegFormer."
        )

    feat = np.stack(
        [
            p_seg,
            s_poi,
            p_base,
            margin,
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    return feat


def candidate_mask(
    gt,
    p_seg,
    s_poi,
    p_base,
    rank,
):
    # p_base < 0.5 means the current SegFormer+POI baseline is still negative.
    # Because the POI stage is positive-only, p_base<0.5 also implies p_seg<0.5,
    # but the explicit p_seg condition is retained to mirror the latest 55b route.
    return (
        (gt != IGNORE_VALUE)
        & (p_seg < THRESHOLD)
        & (p_base < THRESHOLD)
        & (p_base >= VISUAL_FLOOR)
        & (rank == 1)
    )


# =============================================================================
# 8. TRAIN / VAL DATA
# =============================================================================

def build_train_candidates(
    train_ids,
):
    if TRAIN_NPZ.exists():
        z = np.load(
            str(
                TRAIN_NPZ
            )
        )

        print()
        print(
            "TRAIN candidate cache exists:",
            int(
                z[
                    "y"
                ].size
            ),
            "samples"
        )

        return {
            "x": z[
                "x"
            ].astype(
                np.float32
            ),
            "y": z[
                "y"
            ].astype(
                np.float32
            ),
        }

    rng = np.random.default_rng(
        SEED
    )

    xs = []
    ys = []

    print()
    print(
        "Building sampled TRAIN recovery candidates..."
    )

    for i, gid in enumerate(
        train_ids,
        1
    ):
        gt = read_mask(
            gid
        )

        p_seg = read_prob(
            SEG_TRAIN_DIR,
            gid
        )

        p_poi = empirical_school_prior(
            gid
        )

        p_base = read_prob(
            ADAPTIVE_TRAIN_DIR,
            gid
        )

        margin, rank = read_scene(
            SCENE_TRAIN,
            gid
        )

        feat = make_feature_map(
            p_seg,
            p_poi,
            p_base,
            margin,
        )

        cand = candidate_mask(
            gt,
            p_seg,
            p_poi,
            p_base,
            rank,
        )

        pos_idx = np.flatnonzero(
            cand
            & (
                gt
                == 1
            )
        )

        neg_idx = np.flatnonzero(
            cand
            & (
                gt
                == 0
            )
        )

        if pos_idx.size > MAX_POS_PER_TILE:
            pos_idx = rng.choice(
                pos_idx,
                size=MAX_POS_PER_TILE,
                replace=False,
            )

        if pos_idx.size > 0:
            n_neg = min(
                int(
                    neg_idx.size
                ),
                int(
                    math.ceil(
                        NEG_TO_POS_RATIO
                        * pos_idx.size
                    )
                ),
            )
        else:
            n_neg = min(
                int(
                    neg_idx.size
                ),
                NEG_IF_NO_POS,
            )

        if (
            n_neg > 0
            and n_neg
            < neg_idx.size
        ):
            neg_idx = rng.choice(
                neg_idx,
                size=n_neg,
                replace=False,
            )
        elif n_neg == 0:
            neg_idx = np.empty(
                (0,),
                dtype=np.int64
            )

        flat_feat = feat.reshape(
            -1,
            4
        )

        if pos_idx.size > 0:
            xs.append(
                flat_feat[
                    pos_idx
                ]
            )

            ys.append(
                np.ones(
                    (
                        pos_idx.size,
                    ),
                    dtype=np.float32
                )
            )

        if neg_idx.size > 0:
            xs.append(
                flat_feat[
                    neg_idx
                ]
            )

            ys.append(
                np.zeros(
                    (
                        neg_idx.size,
                    ),
                    dtype=np.float32
                )
            )

        if (
            i % 25 == 0
            or i
            == len(
                train_ids
            )
        ):
            current = sum(
                x.shape[
                    0
                ]
                for x in xs
            )

            print(
                "  TRAIN [{}/{}] {} | sampled={}"
                .format(
                    i,
                    len(
                        train_ids
                    ),
                    gid,
                    current,
                )
            )

    if not xs:
        raise RuntimeError(
            "No TRAIN recovery candidates."
        )

    x = np.concatenate(
        xs,
        axis=0
    ).astype(
        np.float32
    )

    y = np.concatenate(
        ys,
        axis=0
    ).astype(
        np.float32
    )

    if x.shape[
        0
    ] > MAX_TRAIN_SAMPLES:
        idx = rng.choice(
            x.shape[
                0
            ],
            size=MAX_TRAIN_SAMPLES,
            replace=False,
        )

        x = x[
            idx
        ]

        y = y[
            idx
        ]

    np.savez_compressed(
        str(
            TRAIN_NPZ
        ),
        x=x,
        y=y,
    )

    print()
    print(
        "TRAIN candidates saved:",
        int(
            y.size
        )
    )

    print(
        "  school={}".format(
            int(
                np.sum(
                    y
                    == 1
                )
            )
        )
    )

    print(
        "  background={}".format(
            int(
                np.sum(
                    y
                    == 0
                )
            )
        )
    )

    return {
        "x": x,
        "y": y,
    }


def build_val_candidates(
    val_ids,
):
    if VAL_NPZ.exists():
        z = np.load(
            str(
                VAL_NPZ
            )
        )

        print()
        print(
            "VAL candidate cache exists:",
            int(
                z[
                    "y"
                ].size
            ),
            "pixels"
        )

        return {
            "x": z[
                "x"
            ].astype(
                np.float32
            ),
            "y": z[
                "y"
            ].astype(
                np.float32
            ),
            "baseline_counts": z[
                "baseline_counts"
            ].astype(
                np.int64
            ),
        }

    xs = []
    ys = []

    baseline = [
        0,
        0,
        0,
        0,
    ]

    print()
    print(
        "Building FULL validation recovery candidates..."
    )

    for i, gid in enumerate(
        val_ids,
        1
    ):
        gt = read_mask(
            gid
        )

        p_seg = read_prob(
            SEG_VAL_DIR,
            gid
        )

        p_poi = empirical_school_prior(
            gid
        )

        p_base = read_prob(
            ADAPTIVE_VAL_DIR,
            gid
        )

        margin, rank = read_scene(
            SCENE_VAL,
            gid
        )

        feat = make_feature_map(
            p_seg,
            p_poi,
            p_base,
            margin,
        )

        base_pred = (
            p_base
            >= THRESHOLD
        ).astype(
            np.uint8
        )

        add_counts(
            baseline,
            counts(
                gt,
                base_pred
            )
        )

        cand = candidate_mask(
            gt,
            p_seg,
            p_poi,
            p_base,
            rank,
        )

        if np.any(
            cand
        ):
            xs.append(
                feat[
                    cand
                ].astype(
                    np.float32
                )
            )

            ys.append(
                (
                    gt[
                        cand
                    ]
                    == 1
                ).astype(
                    np.float32
                )
            )

        if (
            i % 20 == 0
            or i
            == len(
                val_ids
            )
        ):
            current = sum(
                x.shape[
                    0
                ]
                for x in xs
            )

            print(
                "  VAL [{}/{}] {} | candidates={}"
                .format(
                    i,
                    len(
                        val_ids
                    ),
                    gid,
                    current,
                )
            )

    if not xs:
        raise RuntimeError(
            "No VAL recovery candidates."
        )

    x = np.concatenate(
        xs,
        axis=0
    ).astype(
        np.float32
    )

    y = np.concatenate(
        ys,
        axis=0
    ).astype(
        np.float32
    )

    np.savez_compressed(
        str(
            VAL_NPZ
        ),
        x=x,
        y=y,
        baseline_counts=np.asarray(
            baseline,
            dtype=np.int64
        ),
    )

    print()
    print(
        "VAL candidates:",
        int(
            y.size
        )
    )

    print(
        "  school={}".format(
            int(
                np.sum(
                    y
                    == 1
                )
            )
        )
    )

    print(
        "  background={}".format(
            int(
                np.sum(
                    y
                    == 0
                )
            )
        )
    )

    return {
        "x": x,
        "y": y,
        "baseline_counts": np.asarray(
            baseline,
            dtype=np.int64
        ),
    }


def evaluate_validation_reference(
    val_ids,
):
    """
    Hard-check that this VLM experiment starts from the exact completed
    adaptive visual-feature POI validation baseline.
    """
    seg_total = [
        0,
        0,
        0,
        0,
    ]

    base_total = [
        0,
        0,
        0,
        0,
    ]

    for gid in val_ids:
        gt = read_mask(
            gid
        )

        p_seg = read_prob(
            SEG_VAL_DIR,
            gid
        )

        p_base = read_prob(
            ADAPTIVE_VAL_DIR,
            gid
        )

        if np.any(
            p_base
            + 1e-7
            < p_seg
        ):
            raise RuntimeError(
                "Adaptive POI validation map fell below SegFormer: {}"
                .format(
                    gid
                )
            )

        pred_seg = (
            p_seg
            >= THRESHOLD
        ).astype(
            np.uint8
        )

        pred_base = (
            p_base
            >= THRESHOLD
        ).astype(
            np.uint8
        )

        add_counts(
            seg_total,
            counts(
                gt,
                pred_seg,
            ),
        )

        add_counts(
            base_total,
            counts(
                gt,
                pred_base,
            ),
        )

    seg_m = metrics_from_counts(
        *seg_total
    )

    base_m = metrics_from_counts(
        *base_total
    )

    print()
    print(
        "=" * 108
    )
    print(
        "VALIDATION BASELINE CONSISTENCY CHECK"
    )
    print(
        "=" * 108
    )

    print(
        "  SegFormer                   "
        "P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(
            seg_m[
                "precision"
            ],
            seg_m[
                "recall"
            ],
            seg_m[
                "f1"
            ],
            seg_m[
                "iou"
            ],
        )
    )

    print(
        "  Adaptive visual-feature POI "
        "P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(
            base_m[
                "precision"
            ],
            base_m[
                "recall"
            ],
            base_m[
                "f1"
            ],
            base_m[
                "iou"
            ],
        )
    )

    ref = json.loads(
        ADAPTIVE_CACHE_SUMMARY_JSON.read_text(
            encoding="utf-8"
        )
    )

    expected = ref[
        "validation_metrics"
    ]

    for metric in [
        "precision",
        "recall",
        "f1",
        "iou",
        "accuracy",
    ]:
        a = float(
            base_m[
                metric
            ]
        )

        b = float(
            expected[
                metric
            ]
        )

        if abs(
            a - b
        ) > 1e-6:
            raise RuntimeError(
                "Adaptive validation baseline mismatch: "
                "{} current={} expected={}. "
                "Stop before VLM training."
                .format(
                    metric,
                    a,
                    b,
                )
            )

    print(
        "Adaptive-POI validation reference comparison: PASS"
    )

    return (
        seg_m,
        base_m,
    )


# =============================================================================
# 9. RECOVERY GATE
# =============================================================================

class RecoveryGate(
    nn.Module
):
    def __init__(
        self
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                4,
                GATE_H1
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Linear(
                GATE_H1,
                GATE_H2
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Linear(
                GATE_H2,
                1
            ),
        )

    def forward(
        self,
        x
    ):
        return torch.sigmoid(
            self.net(
                x
            ).squeeze(
                1
            )
        )


def compute_normalization(
    x
):
    mean = np.mean(
        x,
        axis=0
    ).astype(
        np.float32
    )

    std = np.std(
        x,
        axis=0
    ).astype(
        np.float32
    )

    std = np.maximum(
        std,
        1e-6
    )

    return (
        mean,
        std,
    )


def normalize_tensor(
    x,
    mean_t,
    std_t,
):
    return (
        x
        - mean_t
    ) / std_t


def recovered_probability(
    raw_x,
    gate,
):
    p_base = raw_x[
        :,
        2
    ]

    return torch.clamp(
        p_base
        + gate
        * (
            1.0
            - p_base
        ),
        0.0,
        1.0
    )


@torch.no_grad()
def evaluate_on_val_candidates(
    model,
    val_data,
    mean,
    std,
    device,
):
    x = val_data[
        "x"
    ]

    y = val_data[
        "y"
    ]

    baseline = [
        int(
            z
        )
        for z in val_data[
            "baseline_counts"
        ]
    ]

    accepted_pos = 0
    accepted_neg = 0

    gate_pos_sum = 0.0
    gate_neg_sum = 0.0
    gate_pos_n = 0
    gate_neg_n = 0

    mean_t = torch.from_numpy(
        mean
    ).to(
        device
    )

    std_t = torch.from_numpy(
        std
    ).to(
        device
    )

    model.eval()

    batch = 65536

    for start in range(
        0,
        x.shape[
            0
        ],
        batch
    ):
        raw = torch.from_numpy(
            x[
                start:
                start
                + batch
            ]
        ).to(
            device
        )

        target = y[
            start:
            start
            + batch
        ]

        z = normalize_tensor(
            raw,
            mean_t,
            std_t,
        )

        g = model(
            z
        )

        p_final = recovered_probability(
            raw,
            g
        )

        accept = (
            p_final
            >= THRESHOLD
        ).cpu().numpy()

        g_np = (
            g.cpu()
            .numpy()
        )

        pos = (
            target
            == 1
        )

        neg = (
            target
            == 0
        )

        accepted_pos += int(
            np.sum(
                accept
                & pos
            )
        )

        accepted_neg += int(
            np.sum(
                accept
                & neg
            )
        )

        if np.any(
            pos
        ):
            gate_pos_sum += float(
                np.sum(
                    g_np[
                        pos
                    ]
                )
            )
            gate_pos_n += int(
                np.sum(
                    pos
                )
            )

        if np.any(
            neg
        ):
            gate_neg_sum += float(
                np.sum(
                    g_np[
                        neg
                    ]
                )
            )
            gate_neg_n += int(
                np.sum(
                    neg
                )
            )

    (
        tp,
        tn,
        fp,
        fn,
    ) = baseline

    # Every candidate is baseline-negative.
    tp2 = (
        tp
        + accepted_pos
    )

    fn2 = (
        fn
        - accepted_pos
    )

    tn2 = (
        tn
        - accepted_neg
    )

    fp2 = (
        fp
        + accepted_neg
    )

    m = metrics_from_counts(
        tp2,
        tn2,
        fp2,
        fn2,
    )

    m.update(
        {
            "recovered_tp": int(
                accepted_pos
            ),
            "added_fp": int(
                accepted_neg
            ),
            "gate_mean_gt1": float(
                gate_pos_sum
                / max(
                    1,
                    gate_pos_n
                )
            ),
            "gate_mean_gt0": float(
                gate_neg_sum
                / max(
                    1,
                    gate_neg_n
                )
            ),
        }
    )

    return m


def train_gate(
    train_data,
    val_data,
    device,
):
    x_train = train_data[
        "x"
    ]

    y_train = train_data[
        "y"
    ]

    mean, std = compute_normalization(
        x_train
    )

    np.save(
        str(
            OUT_ROOT
            / "feature_mean.npy"
        ),
        mean
    )

    np.save(
        str(
            OUT_ROOT
            / "feature_std.npy"
        ),
        std
    )

    dataset = TensorDataset(
        torch.from_numpy(
            x_train
        ).float(),
        torch.from_numpy(
            y_train
        ).float(),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
        drop_last=False,
    )

    model = RecoveryGate().to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    mean_t = torch.from_numpy(
        mean
    ).to(
        device
    )

    std_t = torch.from_numpy(
        std
    ).to(
        device
    )

    baseline_metrics = metrics_from_counts(
        *[
            int(
                z
            )
            for z in val_data[
                "baseline_counts"
            ]
        ]
    )

    # Epoch 0 is EXACT no-op baseline.
    best_epoch = 0
    best_iou = float(
        baseline_metrics[
            "iou"
        ]
    )

    rows = [
        {
            "epoch": 0,
            "train_loss": "",
            "val_precision": baseline_metrics[
                "precision"
            ],
            "val_recall": baseline_metrics[
                "recall"
            ],
            "val_f1": baseline_metrics[
                "f1"
            ],
            "val_iou": baseline_metrics[
                "iou"
            ],
            "val_accuracy": baseline_metrics[
                "accuracy"
            ],
            "recovered_tp": 0,
            "added_fp": 0,
            "gate_mean_gt1": "",
            "gate_mean_gt0": "",
        }
    ]

    torch.save(
        {
            "epoch": 0,
            "noop": True,
            "val_metrics": baseline_metrics,
            "feature_mean": mean,
            "feature_std": std,
        },
        str(
            BEST_CKPT
        )
    )

    print()
    print(
        "Epoch 00 NO-OP baseline | "
        "Val P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(
            baseline_metrics[
                "precision"
            ],
            baseline_metrics[
                "recall"
            ],
            baseline_metrics[
                "f1"
            ],
            baseline_metrics[
                "iou"
            ],
        )
    )

    for epoch in range(
        1,
        EPOCHS
        + 1
    ):
        model.train()

        total_loss = 0.0
        n_batch = 0

        for raw_x, y in loader:
            raw_x = raw_x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            z = normalize_tensor(
                raw_x,
                mean_t,
                std_t,
            )

            g = model(
                z
            )

            p_final = recovered_probability(
                raw_x,
                g
            )

            loss = F.binary_cross_entropy(
                torch.clamp(
                    p_final,
                    EPS,
                    1.0
                    - EPS
                ),
                y
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    "Non-finite loss."
                )

            loss.backward()

            optimizer.step()

            total_loss += float(
                loss.item()
            )

            n_batch += 1

        train_loss = (
            total_loss
            / max(
                1,
                n_batch
            )
        )

        val_m = evaluate_on_val_candidates(
            model,
            val_data,
            mean,
            std,
            device,
        )

        rows.append(
            {
                "epoch": int(
                    epoch
                ),
                "train_loss": float(
                    train_loss
                ),
                "val_precision": val_m[
                    "precision"
                ],
                "val_recall": val_m[
                    "recall"
                ],
                "val_f1": val_m[
                    "f1"
                ],
                "val_iou": val_m[
                    "iou"
                ],
                "val_accuracy": val_m[
                    "accuracy"
                ],
                "recovered_tp": val_m[
                    "recovered_tp"
                ],
                "added_fp": val_m[
                    "added_fp"
                ],
                "gate_mean_gt1": val_m[
                    "gate_mean_gt1"
                ],
                "gate_mean_gt0": val_m[
                    "gate_mean_gt0"
                ],
            }
        )

        print(
            "Epoch {:02d}/{:02d} | loss={:.5f} | "
            "Val P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f} | "
            "recoverTP={} addFP={}"
            .format(
                epoch,
                EPOCHS,
                train_loss,
                val_m[
                    "precision"
                ],
                val_m[
                    "recall"
                ],
                val_m[
                    "f1"
                ],
                val_m[
                    "iou"
                ],
                val_m[
                    "recovered_tp"
                ],
                val_m[
                    "added_fp"
                ],
            )
        )

        if (
            float(
                val_m[
                    "iou"
                ]
            )
            > best_iou
            + 1e-12
        ):
            best_iou = float(
                val_m[
                    "iou"
                ]
            )

            best_epoch = int(
                epoch
            )

            torch.save(
                {
                    "epoch": int(
                        epoch
                    ),
                    "noop": False,
                    "model_state_dict": model.state_dict(),
                    "val_metrics": val_m,
                    "feature_mean": mean,
                    "feature_std": std,
                },
                str(
                    BEST_CKPT
                )
            )

            print(
                "  New BEST recovery gate."
            )

        write_csv(
            TRAIN_LOG_CSV,
            rows
        )

    return (
        best_epoch,
        best_iou,
    )


# =============================================================================
# 10. TEST
# =============================================================================

@torch.no_grad()
def gate_values(
    model,
    raw_features,
    mean,
    std,
    device,
):
    if raw_features.size == 0:
        return np.empty(
            (0,),
            dtype=np.float32
        )

    mean_t = torch.from_numpy(
        mean
    ).to(
        device
    )

    std_t = torch.from_numpy(
        std
    ).to(
        device
    )

    out = np.empty(
        (
            raw_features.shape[
                0
            ],
        ),
        dtype=np.float32
    )

    batch = 262144

    for start in range(
        0,
        raw_features.shape[
            0
        ],
        batch
    ):
        raw = torch.from_numpy(
            raw_features[
                start:
                start
                + batch
            ]
        ).to(
            device
        )

        z = normalize_tensor(
            raw,
            mean_t,
            std_t,
        )

        out[
            start:
            start
            + raw.shape[
                0
            ]
        ] = (
            model(
                z
            )
            .float()
            .cpu()
            .numpy()
        )

    return out


def evaluate_test(
    test_ids,
    ckpt,
    device,
):
    noop = bool(
        ckpt.get(
            "noop",
            False
        )
    )

    mean = np.asarray(
        ckpt[
            "feature_mean"
        ],
        dtype=np.float32
    )

    std = np.asarray(
        ckpt[
            "feature_std"
        ],
        dtype=np.float32
    )

    if not noop:
        model = RecoveryGate().to(
            device
        )

        model.load_state_dict(
            ckpt[
                "model_state_dict"
            ],
            strict=True
        )

        model.eval()

    else:
        model = None

    totals = {
        "SegFormer": [
            0,
            0,
            0,
            0,
        ],
        "Adaptive visual-feature POI": [
            0,
            0,
            0,
            0,
        ],
        "Adaptive visual-feature POI + VLM scene recovery": [
            0,
            0,
            0,
            0,
        ],
    }

    recovered_tp = 0
    added_fp = 0
    candidate_gt1 = 0
    candidate_gt0 = 0

    per_tile = []

    for i, gid in enumerate(
        test_ids,
        1
    ):
        gt = read_mask(
            gid
        )

        p_seg = read_prob(
            SEG_TEST_DIR,
            gid
        )

        p_poi = empirical_school_prior(
            gid
        )

        p_base = read_prob(
            ADAPTIVE_TEST_DIR,
            gid
        )

        margin, rank = read_scene(
            SCENE_TEST,
            gid
        )

        feat = make_feature_map(
            p_seg,
            p_poi,
            p_base,
            margin,
        )

        cand = candidate_mask(
            gt,
            p_seg,
            p_poi,
            p_base,
            rank,
        )

        p_final = p_base.copy()

        gate_map = np.zeros(
            p_base.shape,
            dtype=np.float32
        )

        if (
            (not noop)
            and np.any(
                cand
            )
        ):
            raw = feat[
                cand
            ].astype(
                np.float32
            )

            g = gate_values(
                model,
                raw,
                mean,
                std,
                device,
            )

            base_c = p_base[
                cand
            ]

            p_final[
                cand
            ] = np.clip(
                base_c
                + g
                * (
                    1.0
                    - base_c
                ),
                0.0,
                1.0
            )

            gate_map[
                cand
            ] = g

        pred_seg = (
            p_seg
            >= THRESHOLD
        ).astype(
            np.uint8
        )

        pred_base = (
            p_base
            >= THRESHOLD
        ).astype(
            np.uint8
        )

        pred_final = (
            p_final
            >= THRESHOLD
        ).astype(
            np.uint8
        )

        for name, pred in [
            (
                "SegFormer",
                pred_seg
            ),
            (
                "Adaptive visual-feature POI",
                pred_base
            ),
            (
                "Adaptive visual-feature POI + VLM scene recovery",
                pred_final
            ),
        ]:
            add_counts(
                totals[
                    name
                ],
                counts(
                    gt,
                    pred
                )
            )

        valid = (
            gt
            != IGNORE_VALUE
        )

        new_positive = (
            valid
            & (
                pred_base
                == 0
            )
            & (
                pred_final
                == 1
            )
        )

        tile_recovered_tp = int(
            np.sum(
                new_positive
                & (
                    gt
                    == 1
                )
            )
        )

        tile_added_fp = int(
            np.sum(
                new_positive
                & (
                    gt
                    == 0
                )
            )
        )

        recovered_tp += tile_recovered_tp
        added_fp += tile_added_fp

        candidate_gt1 += int(
            np.sum(
                cand
                & (
                    gt
                    == 1
                )
            )
        )

        candidate_gt0 += int(
            np.sum(
                cand
                & (
                    gt
                    == 0
                )
            )
        )

        per_tile.append(
            {
                "grid_id": gid,
                "candidate_school_pixels": int(
                    np.sum(
                        cand
                        & (
                            gt
                            == 1
                        )
                    )
                ),
                "candidate_background_pixels": int(
                    np.sum(
                        cand
                        & (
                            gt
                            == 0
                        )
                    )
                ),
                "recovered_tp": tile_recovered_tp,
                "added_fp": tile_added_fp,
            }
        )

        np.save(
            str(
                TEST_PROB_DIR
                / (
                    gid
                    + "_school_probability.npy"
                )
            ),
            p_final.astype(
                np.float32
            )
        )

        np.save(
            str(
                TEST_GATE_DIR
                / (
                    gid
                    + "_recovery_gate.npy"
                )
            ),
            gate_map.astype(
                np.float32
            )
        )

        Image.fromarray(
            (
                pred_final
                * 255
            ).astype(
                np.uint8
            ),
            mode="L"
        ).save(
            str(
                TEST_BIN_DIR
                / (
                    gid
                    + "_school.png"
                )
            )
        )

        if (
            i % 10 == 0
            or i
            == len(
                test_ids
            )
        ):
            print(
                "  TEST [{}/{}] {}"
                .format(
                    i,
                    len(
                        test_ids
                    ),
                    gid,
                )
            )

    comparison = []

    metrics_by_method = {}

    for name, total in totals.items():
        m = metrics_from_counts(
            *total
        )

        metrics_by_method[
            name
        ] = m

        comparison.append(
            {
                "method": name,
                **m,
            }
        )

    # Hard-check that the TEST base is exactly the completed adaptive-POI result.
    adaptive_ref = json.loads(
        ADAPTIVE_TEST_SUMMARY_JSON.read_text(
            encoding="utf-8"
        )
    )

    expected_adaptive_test = (
        adaptive_ref[
            "test_metrics"
        ][
            "Adaptive visual-feature POI"
        ]
    )

    current_adaptive_test = (
        metrics_by_method[
            "Adaptive visual-feature POI"
        ]
    )

    for metric in [
        "precision",
        "recall",
        "f1",
        "iou",
        "accuracy",
    ]:
        a = float(
            current_adaptive_test[
                metric
            ]
        )

        b = float(
            expected_adaptive_test[
                metric
            ]
        )

        if abs(
            a - b
        ) > 1e-6:
            raise RuntimeError(
                "Adaptive TEST baseline mismatch: "
                "{} current={} expected={}. "
                "Stop because VLM is not starting from the completed "
                "adaptive-POI result."
                .format(
                    metric,
                    a,
                    b,
                )
            )

    print(
        "Adaptive-POI TEST baseline reference comparison: PASS"
    )

    diag = {
        "selected_epoch": int(
            ckpt[
                "epoch"
            ]
        ),
        "noop_selected": int(
            noop
        ),
        "visual_floor": float(
            VISUAL_FLOOR
        ),
        "vlm_support": (
            "school/campus semantic group rank == 1"
        ),
        "candidate_school_pixels": int(
            candidate_gt1
        ),
        "candidate_background_pixels": int(
            candidate_gt0
        ),
        "recovered_tp": int(
            recovered_tp
        ),
        "added_fp": int(
            added_fp
        ),
        "candidate_school_recovery_rate": float(
            recovered_tp
            / max(
                1,
                candidate_gt1
            )
        ),
        "candidate_background_activation_rate": float(
            added_fp
            / max(
                1,
                candidate_gt0
            )
        ),
    }

    write_csv(
        TEST_DIAG_CSV,
        [
            diag
        ]
    )

    return (
        metrics_by_method,
        diag,
    )


# =============================================================================
# 11. MAIN
# =============================================================================

def main():
    ensure_dirs()
    check_inputs()
    set_seed(
        SEED
    )

    train_ids = read_ids(
        TRAIN_LIST
    )

    val_ids = read_ids(
        VAL_LIST
    )

    test_ids = read_ids(
        TEST_LIST
    )

    if (
        len(train_ids) != 295
        or len(val_ids) != 42
        or len(test_ids) != 84
    ):
        raise RuntimeError(
            "Expected Fold-0 split 295/42/84, got {}/{}/{}."
            .format(len(train_ids), len(val_ids), len(test_ids))
        )

    load_poi_stage_config()

    print(
        "=" * 108
    )
    print(
        "FOLD 0: ADAPTIVE VISUAL-FEATURE POI + VLM SCENE-TRIGGERED RECOVERY"
    )
    print(
        "=" * 108
    )

    print(
        "Base = selected adaptive visual-feature POI probability"
    )

    print(
        "VLM can ONLY add positives; it can never suppress the base."
    )

    print(
        "Candidate: base <0.5, base >= {:.2f}, VLM rank==1"
        .format(
            VISUAL_FLOOR
        )
    )

    text_cfg = build_text_config()

    (
        dgtrs,
        preprocess,
        text_features,
        device,
    ) = load_dgtrs(
        text_cfg
    )

    ensure_scene_maps(
        train_ids,
        "train",
        SCENE_TRAIN,
        dgtrs,
        preprocess,
        text_features,
        device,
        text_cfg,
    )

    ensure_scene_maps(
        val_ids,
        "val",
        SCENE_VAL,
        dgtrs,
        preprocess,
        text_features,
        device,
        text_cfg,
    )

    ensure_scene_maps(
        test_ids,
        "test",
        SCENE_TEST,
        dgtrs,
        preprocess,
        text_features,
        device,
        text_cfg,
    )

    # DGTRS is no longer needed after cached scene maps are ready.
    del dgtrs

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Hard correctness check BEFORE rebuilding candidates/training.
    # This must reproduce the completed adaptive-POI validation baseline.
    (
        val_seg_reference,
        val_union_reference,
    ) = evaluate_validation_reference(
        val_ids
    )

    train_data = build_train_candidates(
        train_ids
    )

    val_data = build_val_candidates(
        val_ids
    )

    val_base = metrics_from_counts(
        *[
            int(
                z
            )
            for z in val_data[
                "baseline_counts"
            ]
        ]
    )

    if abs(
        float(
            val_base[
                "iou"
            ]
        )
        - float(
            val_union_reference[
                "iou"
            ]
        )
    ) > 1e-9:
        raise RuntimeError(
            "Fresh validation candidate cache baseline does not match "
            "the authoritative adaptive-POI validation IoU."
        )

    write_csv(
        VAL_REFERENCE_CSV,
        [
            {
                "method": "Adaptive visual-feature POI baseline",
                **val_base,
            }
        ]
    )

    print()
    print(
        "Validation base:"
    )

    print(
        "  P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(
            val_base[
                "precision"
            ],
            val_base[
                "recall"
            ],
            val_base[
                "f1"
            ],
            val_base[
                "iou"
            ],
        )
    )

    print()
    print(
        "Training recovery gate..."
    )

    # If this current run already completed training and only failed later, resume from its own checkpoint.
    if (
        BEST_CKPT.exists()
        and TRAIN_LOG_CSV.exists()
    ):
        ckpt_meta = torch.load(
            str(
                BEST_CKPT
            ),
            map_location="cpu",
            weights_only=False,
        )

        best_epoch = int(
            ckpt_meta.get(
                "epoch",
                -1
            )
        )

        best_val_iou = float(
            ckpt_meta.get(
                "val_metrics",
                {}
            ).get(
                "iou",
                float(
                    "nan"
                )
            )
        )

        print(
            "Current VLM checkpoint found; SKIPPING retraining."
        )
        print(
            "  best epoch =",
            best_epoch
        )
        print(
            "  best val IoU =",
            best_val_iou
        )

    else:
        best_epoch, best_val_iou = train_gate(
            train_data,
            val_data,
            device,
        )

    # PyTorch >=2.6 defaults weights_only=True. This locally generated
    # checkpoint intentionally contains NumPy feature_mean/std and metadata.
    ckpt = torch.load(
        str(
            BEST_CKPT
        ),
        map_location="cpu",
        weights_only=False,
    )

    print()
    print(
        "Selected epoch:",
        best_epoch
    )

    print(
        "Best validation IoU:",
        best_val_iou
    )

    print()
    print(
        "=" * 108
    )
    print(
        "HELD-OUT TEST: ONCE"
    )
    print(
        "=" * 108
    )

    metrics_by_method, diag = evaluate_test(
        test_ids,
        ckpt,
        device,
    )

    print()
    print(
        "{:<42s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"
        .format(
            "METHOD",
            "P",
            "R",
            "F1",
            "IoU",
            "ACC"
        )
    )

    for name in [
        "SegFormer",
        "Adaptive visual-feature POI",
        "Adaptive visual-feature POI + VLM scene recovery",
    ]:
        m = metrics_by_method[
            name
        ]

        print(
            "{:<42s} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}"
            .format(
                name,
                m[
                    "precision"
                ],
                m[
                    "recall"
                ],
                m[
                    "f1"
                ],
                m[
                    "iou"
                ],
                m[
                    "accuracy"
                ],
            )
        )

    print()
    print(
        "Recovery diagnostic:"
    )

    print(
        "  candidate school pixels     :",
        diag[
            "candidate_school_pixels"
        ]
    )

    print(
        "  candidate background pixels :",
        diag[
            "candidate_background_pixels"
        ]
    )

    print(
        "  recovered TP                :",
        diag[
            "recovered_tp"
        ]
    )

    print(
        "  added FP                    :",
        diag[
            "added_fp"
        ]
    )

    print(
        "  school recovery rate        : {:.4f}"
        .format(
            diag[
                "candidate_school_recovery_rate"
            ]
        )
    )

    print(
        "  background activation rate  : {:.4f}"
        .format(
            diag[
                "candidate_background_activation_rate"
            ]
        )
    )

    summary = {
        "diagnostic_fold0": True,
        "formal_final_experiment_required": True,
        "poi_radius_primary_m": float(EMPIRICAL_RADIUS_M["primary"]),
        "poi_radius_middle_m": float(EMPIRICAL_RADIUS_M["middle"]),
        "poi_radius_higher_m": float(EMPIRICAL_RADIUS_M["higher"]),
        "fixed_poi_reference_lambda": float(SELECTED_POI_LAMBDA),
        "method": (
            "selected visual-feature adaptive POI base with DGTRS/LRSCLIP "
            "scene-triggered recovery gate"
        ),
        "base_formula": (
            "P_base = frozen selected output of the visual-feature adaptive "
            "POI Gate; fixed global-lambda POI is replaced, not appended"
        ),
        "recovery_formula": (
            "P_final=P_base+g*(1-P_base) on candidate pixels only"
        ),
        "monotonicity": (
            "P_final>=P_base; VLM branch cannot suppress existing positives"
        ),
        "candidate_definition": (
            "valid core AND P_seg<0.5 AND P_base<0.5 AND "
            "P_base>=0.20 AND VLM school/campus group rank==1"
        ),
        "gate_inputs": [
            "P_seg",
            "S_poi",
            "P_base",
            "VLM_scene_margin",
        ],
        "seg_poi_threshold": THRESHOLD,
        "visual_floor": VISUAL_FLOOR,
        "selected_epoch": int(
            best_epoch
        ),
        "best_val_iou": float(
            best_val_iou
        ),
        "test_metrics": metrics_by_method,
        "test_recovery_diagnostic": diag,
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    print()
    print(
        "Output:",
        OUT_ROOT
    )


if __name__ == "__main__":
    main()
