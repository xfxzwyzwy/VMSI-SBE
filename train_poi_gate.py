# -*- coding: utf-8 -*-
r"""
Fold 0: visual-feature adaptive POI fusion
==========================================

Purpose
-------
Directly compare:

    A) current fixed SegFormer + POI
       P_fixed = sigmoid(logit(P_seg) + lambda_ref * S_poi)

vs.

    B) visual-feature adaptive SegFormer + POI
       lambda(x) = 2 * lambda_ref * sigmoid(Gate(...))
       P_adapt   = sigmoid(logit(P_seg) + lambda(x) * S_poi)

The POI spatial ranges are NOT changed.
The current empirical radii are read from the completed POI stage:
    primary / 小学
    middle  / 中学
    higher  / 高等教育

The current global lambda_ref is also read from the completed POI stage.
For the current Fold-0 diagnostic this should be:
    lambda_ref = 0.40

Only the scalar fusion strength is replaced by a pixelwise adaptive map.

Why SegFormer visual features are used
--------------------------------------
The Gate receives:
    1) frozen SegFormer decoder feature
    2) frozen SegFormer probability P_seg
    3) fixed empirical-radius POI prior S_poi

The SegFormer decoder feature is captured immediately before the frozen final
classifier, exactly following the earlier visual-feature Gate idea.

The Gate learns where POI evidence should be stronger or weaker:

    lambda(x) > lambda_ref  -> POI contributes more than the fixed baseline
    lambda(x) < lambda_ref  -> POI contributes less than the fixed baseline

Because lambda(x) >= 0:
    P_adapt >= P_seg

Therefore the adaptive POI branch still cannot suppress the original
SegFormer probability below the pure visual baseline.

Initialization
--------------
The Gate output layer is initialized so:

    sigmoid(output) = 0.5
    lambda(x) = lambda_ref

everywhere at epoch 0.

Therefore epoch 0 is EXACTLY the current fixed SegFormer + POI method
(up to floating-point tolerance).

This is important: the experiment is a clean test of whether the frozen
SegFormer visual feature improves the already-completed fixed POI fusion.

Training / selection
--------------------
Frozen:
    - SegFormer-B2
    - empirical POI radii
    - all-school POI cache
    - current lambda_ref

Trainable:
    - visual projection + tiny Gate only

Training:
    - Fold-0 TRAIN only
    - same 512 crop size
    - 2 crops per TRAIN image per epoch
    - school-centered crop probability = 0.65
    - flip / rotate augmentation
    - 20 epochs
    - AdamW, lr=2e-4
    - batch size 1

Loss:
    L = L_final + 0.5 * L_gate_pref

L_final:
    NLL on the final adaptive probability.

L_gate_pref:
    only pixels with S_poi > 1e-4 are informative.
    GT school     -> prefer stronger POI weight (gate -> 1)
    GT background -> prefer weaker POI weight (gate -> 0)
    school/background preference groups are balanced.

Validation:
    - epoch 0 included as exact fixed-POI no-op reference
    - every 5 epochs
    - best checkpoint selected by full validation-core micro IoU

Test:
    - held-out TEST evaluated once after checkpoint selection
    - threshold fixed at 0.5

Important
---------
This script is still a Fold-0 diagnostic because the empirical radii used by
the current POI stage were obtained from the previous all-sample descriptive
analysis. If this route is retained formally, each fold's radii should be
estimated using TRAIN buildings only.

Environment
-----------
Use the normal SegFormer environment:
    conda activate larse

No rasterio is used.

Outputs
-------
E:\larse_result\segformer_poi_visualfeature_gate_fold0\

    checkpoints\
        best_gate.pth
        last_gate.pth

    training_log.csv
    selected_config.json
    validation_reference.json

    test_summary.json
    test_metrics_by_school_type.csv
    comparison_with_fixed_poi.csv

    test_predictions\
        adaptive_probability_npy\
        lambda_map_npy\
        binary_png\
"""

from __future__ import print_function

import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError:
    raise RuntimeError(
        "Missing transformers. Run this script in the SegFormer environment."
    )


# =============================================================================
# 1. PATHS
# =============================================================================

DATASET_ROOT = Path(
    r"G:\SH\上海市\学校实验区_500m_B150m\0924\segformer_binary_5fold_dataset"
)

IMAGE_DIR = DATASET_ROOT / "images"
MASK_DIR = DATASET_ROOT / "masks_png"
TYPE_MASK_DIR = DATASET_ROOT / "masks_type_png"
FOLD_DIR = DATASET_ROOT / "folds" / "fold_0"

PRETRAINED_MODEL = (
    Path(__file__).resolve().parent / "pretrained" / "segformer_b2"
)

SEGFORMER_FOLD_ROOT = Path(
    r"E:\larse_result\segformer_b2_binary_5fold\fold_0"
)

SEGFORMER_BEST_CKPT = (
    SEGFORMER_FOLD_ROOT
    / "checkpoints"
    / "best_model.pth"
)

TRAIN_SEG_PROB_DIR = Path(
    r"E:\larse_result\segformer_poi_competitive_5fold\fold_0"
    r"\train_visual_probabilities\school_probability_npy"
)

VAL_SEG_PROB_DIR = (
    SEGFORMER_FOLD_ROOT
    / "val_predictions_best"
    / "school_probability_npy"
)

TEST_SEG_PROB_DIR = (
    SEGFORMER_FOLD_ROOT
    / "test_predictions"
    / "school_probability_npy"
)

ALL_SCHOOL_CACHE_DIR = Path(
    r"E:\larse_result\poi_all_school_base_prior_cache\cache"
)

FIXED_POI_ROOT = Path(
    r"E:\larse_result\segformer_poi_empirical_radius_logit_fold0"
)

FIXED_POI_CONFIG = (
    FIXED_POI_ROOT
    / "selected_config.json"
)

FIXED_POI_TEST_PROB_DIR = (
    FIXED_POI_ROOT
    / "test_predictions"
    / "school_probability_npy"
)

OUT_ROOT = Path(
    r"E:\larse_result\segformer_poi_visualfeature_gate_fold0"
)

CKPT_DIR = OUT_ROOT / "checkpoints"
BEST_CKPT = CKPT_DIR / "best_gate.pth"
LAST_CKPT = CKPT_DIR / "last_gate.pth"

TRAIN_LOG = OUT_ROOT / "training_log.csv"
SELECTED_CONFIG = OUT_ROOT / "selected_config.json"
VAL_REFERENCE = OUT_ROOT / "validation_reference.json"
TEST_SUMMARY = OUT_ROOT / "test_summary.json"
TEST_TYPE_CSV = OUT_ROOT / "test_metrics_by_school_type.csv"
TEST_COMPARE_CSV = OUT_ROOT / "comparison_with_fixed_poi.csv"

TEST_OUT = OUT_ROOT / "test_predictions"
TEST_ADAPT_PROB_DIR = TEST_OUT / "adaptive_probability_npy"
TEST_LAMBDA_DIR = TEST_OUT / "lambda_map_npy"
TEST_BINARY_DIR = TEST_OUT / "binary_png"


# =============================================================================
# 2. SETTINGS
# =============================================================================

SEED = 42

CROP_SIZE = 512
INFER_STRIDE = 256

BATCH_SIZE = 1
EPOCHS = 20
VAL_EVERY = 5
PATCHES_PER_IMAGE_PER_EPOCH = 2
SCHOOL_CENTER_PROB = 0.65

GATE_LR = 2e-4
WEIGHT_DECAY = 0.0

GATE_VISUAL_FEATURE_CHANNELS = 8

# Auxiliary gate supervision, adapted directly from the old visual-feature Gate.
LAMBDA_GATE_PREF = 0.50
GATE_INFORMATIVE_EPS = 1e-4

THRESHOLD = 0.50
IGNORE_VALUE = 255
EPS = 1e-6

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)

SCHOOL_TYPES = [
    "primary",
    "middle",
    "higher",
]

TYPE_TO_CACHE_INDEX = {
    "primary": 0,
    "middle": 1,
    "higher": 2,
}

TYPE_TO_MASK_VALUE = {
    "primary": 1,
    "middle": 2,
    "higher": 3,
}

TYPE_TO_CN = {
    "primary": "小学",
    "middle": "中学",
    "higher": "高等教育",
}

FOUR_SIGMA_CUTOFF = float(
    math.exp(-8.0)
)

# Filled from the completed fixed-POI stage.
EMPIRICAL_RADIUS_M = None
LAMBDA_REF = None


# =============================================================================
# 3. GENERAL HELPERS
# =============================================================================

def ensure_dirs():
    for p in [
        OUT_ROOT,
        CKPT_DIR,
        TEST_OUT,
        TEST_ADAPT_PROB_DIR,
        TEST_LAMBDA_DIR,
        TEST_BINARY_DIR,
    ]:
        p.mkdir(
            parents=True,
            exist_ok=True,
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_ids(path):
    with open(
        str(path),
        "r",
        encoding="utf-8",
    ) as f:
        return [
            x.strip()
            for x in f
            if x.strip()
        ]


def find_file(root, gid, suffixes):
    root = Path(root)

    for ext in suffixes:
        p = root / (str(gid) + ext)

        if p.exists():
            return p

    raise FileNotFoundError(
        "Cannot find {} in {}".format(
            gid,
            root,
        )
    )


def read_rgb(gid):
    p = find_file(
        IMAGE_DIR,
        gid,
        [".tif", ".tiff", ".png", ".jpg", ".jpeg"],
    )

    with Image.open(
        str(p)
    ) as im:
        return np.asarray(
            im.convert("RGB"),
            dtype=np.uint8,
        )


def read_mask(gid):
    p = MASK_DIR / (str(gid) + ".png")

    with Image.open(
        str(p)
    ) as im:
        return np.asarray(
            im,
            dtype=np.uint8,
        )


def read_type_mask(gid):
    p = TYPE_MASK_DIR / (str(gid) + ".png")

    with Image.open(
        str(p)
    ) as im:
        return np.asarray(
            im,
            dtype=np.uint8,
        )


def probability_path(root, gid):
    root = Path(root)

    candidates = [
        root / (str(gid) + "_school_probability.npy"),
        root / (str(gid) + ".npy"),
    ]

    for p in candidates:
        if p.exists():
            return p

    return candidates[0]


def read_probability(root, gid):
    p = probability_path(
        root,
        gid,
    )

    if not p.exists():
        raise FileNotFoundError(p)

    arr = np.load(
        str(p)
    ).astype(
        np.float32
    )

    return np.clip(
        arr,
        0.0,
        1.0,
    )


def image_to_tensor(rgb):
    x = (
        rgb.astype(
            np.float32
        )
        / 255.0
    )

    x = (
        x
        - IMAGENET_MEAN
    ) / IMAGENET_STD

    return torch.from_numpy(
        np.ascontiguousarray(
            x.transpose(
                2,
                0,
                1,
            )
        )
    ).float()


def write_json(path, obj):
    with open(
        str(path),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=2,
        )


def check_inputs():
    required = [
        DATASET_ROOT,
        IMAGE_DIR,
        MASK_DIR,
        TYPE_MASK_DIR,
        FOLD_DIR,
        PRETRAINED_MODEL,
        SEGFORMER_BEST_CKPT,
        TRAIN_SEG_PROB_DIR,
        VAL_SEG_PROB_DIR,
        TEST_SEG_PROB_DIR,
        ALL_SCHOOL_CACHE_DIR,
        FIXED_POI_CONFIG,
    ]

    for p in required:
        if not Path(p).exists():
            raise FileNotFoundError(
                "Required input not found:\n{}".format(p)
            )


# =============================================================================
# 4. LOAD THE COMPLETED FIXED-POI CONFIG
# =============================================================================

def load_fixed_poi_config():
    global EMPIRICAL_RADIUS_M
    global LAMBDA_REF

    cfg = json.loads(
        FIXED_POI_CONFIG.read_text(
            encoding="utf-8"
        )
    )

    EMPIRICAL_RADIUS_M = {
        "primary": float(
            cfg[
                "radius_primary_m"
            ]
        ),
        "middle": float(
            cfg[
                "radius_middle_m"
            ]
        ),
        "higher": float(
            cfg[
                "radius_higher_m"
            ]
        ),
    }

    LAMBDA_REF = float(
        cfg[
            "selected_lambda"
        ]
    )

    if LAMBDA_REF <= 0.0:
        raise RuntimeError(
            "Expected positive fixed POI lambda, got {}"
            .format(
                LAMBDA_REF
            )
        )

    print(
        "Loaded completed fixed-POI stage:"
    )

    print(
        "  primary={} m, middle={} m, higher={} m, lambda_ref={:.4f}"
        .format(
            EMPIRICAL_RADIUS_M[
                "primary"
            ],
            EMPIRICAL_RADIUS_M[
                "middle"
            ],
            EMPIRICAL_RADIUS_M[
                "higher"
            ],
            LAMBDA_REF,
        )
    )


# =============================================================================
# 5. EMPIRICAL-RADIUS POI PRIOR
# =============================================================================

def load_school_base(gid):
    p = (
        ALL_SCHOOL_CACHE_DIR
        / (
            str(gid)
            + ".npz"
        )
    )

    if not p.exists():
        raise FileNotFoundError(p)

    with np.load(
        str(p),
        allow_pickle=False,
    ) as z:
        base = z[
            "school_base"
        ].astype(
            np.float32
        )

        base_sigma = float(
            z[
                "base_sigma_m"
            ][0]
        )

    if (
        base.ndim != 3
        or base.shape[0] != 3
    ):
        raise RuntimeError(
            "{}: school_base shape={} expected [3,H,W]"
            .format(
                gid,
                base.shape,
            )
        )

    return (
        base,
        base_sigma,
    )


def base_gaussian_to_distance(
    base_prior,
    base_sigma_m,
):
    base = np.clip(
        base_prior.astype(
            np.float64,
            copy=False,
        ),
        0.0,
        1.0,
    )

    d = np.full(
        base.shape,
        np.inf,
        dtype=np.float32,
    )

    positive = (
        base > 0.0
    )

    if np.any(
        positive
    ):
        x = np.maximum(
            base[
                positive
            ],
            1e-300,
        )

        dist = (
            float(
                base_sigma_m
            )
            * np.sqrt(
                np.maximum(
                    0.0,
                    -2.0
                    * np.log(x),
                )
            )
        )

        d[
            positive
        ] = dist.astype(
            np.float32
        )

    return d


def cosine_prior_from_distance(
    distance_m,
    radius_m,
):
    out = np.zeros(
        distance_m.shape,
        dtype=np.float32,
    )

    inside = (
        distance_m
        <= float(
            radius_m
        )
    )

    if np.any(
        inside
    ):
        x = (
            distance_m[
                inside
            ].astype(
                np.float64
            )
            / float(
                radius_m
            )
        )

        out[
            inside
        ] = (
            0.5
            * (
                1.0
                + np.cos(
                    np.pi
                    * x
                )
            )
        ).astype(
            np.float32
        )

    return out


def empirical_school_prior(gid):
    base, base_sigma = (
        load_school_base(
            gid
        )
    )

    priors = []

    for name in SCHOOL_TYPES:
        idx = (
            TYPE_TO_CACHE_INDEX[
                name
            ]
        )

        d = base_gaussian_to_distance(
            base[
                idx
            ],
            base_sigma,
        )

        p = cosine_prior_from_distance(
            d,
            EMPIRICAL_RADIUS_M[
                name
            ],
        )

        priors.append(
            p
        )

    return np.maximum.reduce(
        priors
    ).astype(
        np.float32
    )


# =============================================================================
# 6. PROBABILITY HELPERS / METRICS
# =============================================================================

def np_logit(p):
    p = np.clip(
        p,
        EPS,
        1.0 - EPS,
    )

    return np.log(
        p
        / (
            1.0 - p
        )
    )


def np_sigmoid(z):
    z = np.asarray(
        z,
        dtype=np.float64,
    )

    out = np.empty_like(
        z,
        dtype=np.float64,
    )

    pos = (
        z >= 0
    )

    out[
        pos
    ] = (
        1.0
        / (
            1.0
            + np.exp(
                -z[
                    pos
                ]
            )
        )
    )

    ez = np.exp(
        z[
            ~pos
        ]
    )

    out[
        ~pos
    ] = (
        ez
        / (
            1.0 + ez
        )
    )

    return out.astype(
        np.float32
    )


def fixed_poi_probability(
    p_seg,
    s_poi,
):
    return np_sigmoid(
        np_logit(
            p_seg
        )
        + float(
            LAMBDA_REF
        )
        * s_poi
    )


def counts_from_prob(
    gt,
    prob,
):
    pred = (
        prob
        >= THRESHOLD
    ).astype(
        np.uint8
    )

    valid = (
        gt
        != IGNORE_VALUE
    )

    g = gt[
        valid
    ]

    p = pred[
        valid
    ]

    tp = int(
        np.sum(
            (g == 1)
            & (
                p == 1
            )
        )
    )

    tn = int(
        np.sum(
            (g == 0)
            & (
                p == 0
            )
        )
    )

    fp = int(
        np.sum(
            (g == 0)
            & (
                p == 1
            )
        )
    )

    fn = int(
        np.sum(
            (g == 1)
            & (
                p == 0
            )
        )
    )

    return [
        tp,
        tn,
        fp,
        fn,
    ]


def add_counts(
    total,
    c,
):
    for i in range(4):
        total[
            i
        ] += int(
            c[
                i
            ]
        )


def metrics_from_counts(c):
    tp, tn, fp, fn = c

    eps = 1e-12

    return {
        "precision": float(
            tp
            / (
                tp
                + fp
                + eps
            )
        ),
        "recall": float(
            tp
            / (
                tp
                + fn
                + eps
            )
        ),
        "f1": float(
            2.0
            * tp
            / (
                2.0
                * tp
                + fp
                + fn
                + eps
            )
        ),
        "iou": float(
            tp
            / (
                tp
                + fp
                + fn
                + eps
            )
        ),
        "accuracy": float(
            (
                tp + tn
            )
            / (
                tp
                + tn
                + fp
                + fn
                + eps
            )
        ),
        "TP": int(
            tp
        ),
        "TN": int(
            tn
        ),
        "FP": int(
            fp
        ),
        "FN": int(
            fn
        ),
    }


def type_counts(
    type_gt,
    prob,
    target_value,
):
    if not np.any(
        type_gt
        == int(
            target_value
        )
    ):
        return None

    valid = (
        (
            type_gt
            == 0
        )
        | (
            type_gt
            == int(
                target_value
            )
        )
    )

    g = (
        type_gt[
            valid
        ]
        == int(
            target_value
        )
    ).astype(
        np.uint8
    )

    p = (
        prob[
            valid
        ]
        >= THRESHOLD
    ).astype(
        np.uint8
    )

    tp = int(
        np.sum(
            (g == 1)
            & (
                p == 1
            )
        )
    )

    tn = int(
        np.sum(
            (g == 0)
            & (
                p == 0
            )
        )
    )

    fp = int(
        np.sum(
            (g == 0)
            & (
                p == 1
            )
        )
    )

    fn = int(
        np.sum(
            (g == 1)
            & (
                p == 0
            )
        )
    )

    return [
        tp,
        tn,
        fp,
        fn,
    ]


# =============================================================================
# 7. TRAIN CROP DATASET
# =============================================================================

def pad_to_crop(
    rgb,
    mask,
    p_seg,
    s_poi,
):
    h, w = mask.shape

    pad_h = max(
        0,
        CROP_SIZE - h,
    )

    pad_w = max(
        0,
        CROP_SIZE - w,
    )

    if (
        pad_h == 0
        and pad_w == 0
    ):
        return (
            rgb,
            mask,
            p_seg,
            s_poi,
        )

    rgb = np.pad(
        rgb,
        (
            (
                0,
                pad_h,
            ),
            (
                0,
                pad_w,
            ),
            (
                0,
                0,
            ),
        ),
        mode="edge",
    )

    mask = np.pad(
        mask,
        (
            (
                0,
                pad_h,
            ),
            (
                0,
                pad_w,
            ),
        ),
        mode="constant",
        constant_values=IGNORE_VALUE,
    )

    p_seg = np.pad(
        p_seg,
        (
            (
                0,
                pad_h,
            ),
            (
                0,
                pad_w,
            ),
        ),
        mode="edge",
    )

    s_poi = np.pad(
        s_poi,
        (
            (
                0,
                pad_h,
            ),
            (
                0,
                pad_w,
            ),
        ),
        mode="constant",
        constant_values=0.0,
    )

    return (
        rgb,
        mask,
        p_seg,
        s_poi,
    )


class GateCropDataset(Dataset):

    def __init__(
        self,
        grid_ids,
        cache_size=8,
    ):
        self.grid_ids = list(
            grid_ids
        )

        self.cache_size = int(
            cache_size
        )

        self.cache = OrderedDict()

    def __len__(self):
        return (
            len(
                self.grid_ids
            )
            * PATCHES_PER_IMAGE_PER_EPOCH
        )

    def _load(
        self,
        gid,
    ):
        if gid in self.cache:
            item = self.cache.pop(
                gid
            )

            self.cache[
                gid
            ] = item

            return item

        rgb = read_rgb(
            gid
        )

        mask = read_mask(
            gid
        )

        p_seg = read_probability(
            TRAIN_SEG_PROB_DIR,
            gid,
        )

        s_poi = empirical_school_prior(
            gid
        )

        if not (
            rgb.shape[
                :2
            ]
            == mask.shape
            == p_seg.shape
            == s_poi.shape
        ):
            raise RuntimeError(
                "{} shape mismatch: rgb={} mask={} pseg={} poi={}"
                .format(
                    gid,
                    rgb.shape,
                    mask.shape,
                    p_seg.shape,
                    s_poi.shape,
                )
            )

        item = pad_to_crop(
            rgb,
            mask,
            p_seg,
            s_poi,
        )

        self.cache[
            gid
        ] = item

        while (
            len(
                self.cache
            )
            > self.cache_size
        ):
            self.cache.popitem(
                last=False
            )

        return item

    def __getitem__(
        self,
        index,
    ):
        gid = self.grid_ids[
            index
            % len(
                self.grid_ids
            )
        ]

        (
            rgb,
            mask,
            p_seg,
            s_poi,
        ) = self._load(
            gid
        )

        h, w = mask.shape

        sy, sx = np.where(
            mask == 1
        )

        vy, vx = np.where(
            mask
            != IGNORE_VALUE
        )

        use_school = (
            len(
                sy
            ) > 0
            and random.random()
            < SCHOOL_CENTER_PROB
        )

        if use_school:
            j = random.randrange(
                len(
                    sy
                )
            )

            cy = int(
                sy[
                    j
                ]
            )

            cx = int(
                sx[
                    j
                ]
            )

        elif len(
            vy
        ) > 0:
            j = random.randrange(
                len(
                    vy
                )
            )

            cy = int(
                vy[
                    j
                ]
            )

            cx = int(
                vx[
                    j
                ]
            )

        else:
            cy = h // 2
            cx = w // 2

        y0 = int(
            np.clip(
                cy
                - CROP_SIZE
                // 2,
                0,
                h
                - CROP_SIZE,
            )
        )

        x0 = int(
            np.clip(
                cx
                - CROP_SIZE
                // 2,
                0,
                w
                - CROP_SIZE,
            )
        )

        rgb_c = rgb[
            y0:
            y0
            + CROP_SIZE,
            x0:
            x0
            + CROP_SIZE,
        ]

        mask_c = mask[
            y0:
            y0
            + CROP_SIZE,
            x0:
            x0
            + CROP_SIZE,
        ]

        pseg_c = p_seg[
            y0:
            y0
            + CROP_SIZE,
            x0:
            x0
            + CROP_SIZE,
        ]

        poi_c = s_poi[
            y0:
            y0
            + CROP_SIZE,
            x0:
            x0
            + CROP_SIZE,
        ]

        # Same geometric transform applied to all four arrays.
        if random.random() < 0.5:
            rgb_c = np.flip(
                rgb_c,
                axis=1,
            )

            mask_c = np.flip(
                mask_c,
                axis=1,
            )

            pseg_c = np.flip(
                pseg_c,
                axis=1,
            )

            poi_c = np.flip(
                poi_c,
                axis=1,
            )

        if random.random() < 0.5:
            rgb_c = np.flip(
                rgb_c,
                axis=0,
            )

            mask_c = np.flip(
                mask_c,
                axis=0,
            )

            pseg_c = np.flip(
                pseg_c,
                axis=0,
            )

            poi_c = np.flip(
                poi_c,
                axis=0,
            )

        k = random.randint(
            0,
            3,
        )

        if k:
            rgb_c = np.rot90(
                rgb_c,
                k=k,
                axes=(
                    0,
                    1,
                ),
            )

            mask_c = np.rot90(
                mask_c,
                k=k,
                axes=(
                    0,
                    1,
                ),
            )

            pseg_c = np.rot90(
                pseg_c,
                k=k,
                axes=(
                    0,
                    1,
                ),
            )

            poi_c = np.rot90(
                poi_c,
                k=k,
                axes=(
                    0,
                    1,
                ),
            )

        x = image_to_tensor(
            np.ascontiguousarray(
                rgb_c
            )
        )

        y = torch.from_numpy(
            np.ascontiguousarray(
                mask_c
            ).astype(
                np.int64
            )
        ).long()

        pseg_t = torch.from_numpy(
            np.ascontiguousarray(
                pseg_c
            )
        ).float().unsqueeze(
            0
        )

        poi_t = torch.from_numpy(
            np.ascontiguousarray(
                poi_c
            )
        ).float().unsqueeze(
            0
        )

        return (
            x,
            y,
            pseg_t,
            poi_t,
        )


# =============================================================================
# 8. FROZEN SEGFORMER + VISUAL-FEATURE GATE
# =============================================================================

def build_frozen_segformer(
    device,
):
    print(
        "Loading frozen SegFormer-B2..."
    )

    model = (
        SegformerForSemanticSegmentation
        .from_pretrained(
            str(
                PRETRAINED_MODEL
            ),
            num_labels=2,
            id2label={
                0: "background",
                1: "school",
            },
            label2id={
                "background": 0,
                "school": 1,
            },
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
    )

    obj = torch.load(
        str(
            SEGFORMER_BEST_CKPT
        ),
        map_location="cpu",
    )

    model.load_state_dict(
        obj[
            "model_state_dict"
        ],
        strict=True,
    )

    for p in model.parameters():
        p.requires_grad = False

    model = model.to(
        device
    )

    model.eval()

    return model


class PixelwiseLambdaGate(nn.Module):

    def __init__(
        self,
        decoder_channels,
        lambda_ref,
    ):
        super().__init__()

        self.lambda_ref = float(
            lambda_ref
        )

        self.visual_projection = nn.Sequential(
            nn.Conv2d(
                int(
                    decoder_channels
                ),
                GATE_VISUAL_FEATURE_CHANNELS,
                kernel_size=1,
                bias=False,
            ),
            nn.GroupNorm(
                4,
                GATE_VISUAL_FEATURE_CHANNELS,
            ),
            nn.ReLU(
                inplace=True
            ),
        )

        # Inputs:
        # visual feature + P_seg + S_poi
        in_channels = (
            GATE_VISUAL_FEATURE_CHANNELS
            + 1
            + 1
        )

        self.body = nn.Sequential(
            nn.Conv2d(
                in_channels,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                4,
                16,
            ),
            nn.ReLU(
                inplace=True
            ),
            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                4,
                16,
            ),
            nn.ReLU(
                inplace=True
            ),
        )

        self.out = nn.Conv2d(
            16,
            1,
            kernel_size=1,
            bias=True,
        )

        # Epoch 0:
        # sigmoid(0)=0.5
        # lambda_map=2*lambda_ref*0.5=lambda_ref exactly.
        nn.init.zeros_(
            self.out.weight
        )

        nn.init.zeros_(
            self.out.bias
        )

    def forward(
        self,
        decoder_feature,
        p_seg,
        s_poi,
    ):
        visual = self.visual_projection(
            decoder_feature
        )

        if (
            visual.shape[
                -2:
            ]
            != p_seg.shape[
                -2:
            ]
        ):
            visual = F.interpolate(
                visual,
                size=p_seg.shape[
                    -2:
                ],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [
                visual,
                p_seg,
                s_poi,
            ],
            dim=1,
        )

        gate = torch.sigmoid(
            self.out(
                self.body(
                    x
                )
            )
        )

        lambda_map = (
            2.0
            * self.lambda_ref
            * gate
        )

        return (
            gate,
            lambda_map,
        )


class FrozenSegVisualLambdaNet(nn.Module):

    def __init__(
        self,
        segformer,
        lambda_ref,
    ):
        super().__init__()

        self.segformer = segformer
        self.lambda_ref = float(
            lambda_ref
        )

        classifier = (
            self.segformer
            .decode_head
            .classifier
        )

        decoder_channels = int(
            classifier.in_channels
        )

        self.gate = PixelwiseLambdaGate(
            decoder_channels=decoder_channels,
            lambda_ref=lambda_ref,
        )

        self._last_decoder_feature = None

        def capture_feature(
            module,
            inputs,
        ):
            if not inputs:
                raise RuntimeError(
                    "SegFormer classifier pre-hook received no input."
                )

            feat = inputs[
                0
            ]

            if not torch.is_tensor(
                feat
            ):
                raise RuntimeError(
                    "Unexpected decoder feature type: {}"
                    .format(
                        type(
                            feat
                        )
                    )
                )

            self._last_decoder_feature = feat.detach()

        self._hook = (
            classifier
            .register_forward_pre_hook(
                capture_feature
            )
        )

        print(
            "Visual Gate input: frozen SegFormer decoder feature "
            "({} channels) -> {} projected channels."
            .format(
                decoder_channels,
                GATE_VISUAL_FEATURE_CHANNELS,
            )
        )

        print(
            "lambda(x) range: [0, {:.4f}], epoch-0 reference={:.4f}"
            .format(
                2.0
                * self.lambda_ref,
                self.lambda_ref,
            )
        )

    def train(
        self,
        mode=True,
    ):
        super().train(
            mode
        )

        # SegFormer always stays frozen/eval.
        self.segformer.eval()

        return self

    def decoder_feature(
        self,
        pixel_values,
    ):
        self._last_decoder_feature = None

        with torch.no_grad():
            _ = self.segformer(
                pixel_values=pixel_values
            )

        feat = self._last_decoder_feature

        if feat is None:
            raise RuntimeError(
                "Failed to capture SegFormer decoder feature."
            )

        return feat

    def forward(
        self,
        pixel_values,
        p_seg,
        s_poi,
    ):
        feat = self.decoder_feature(
            pixel_values
        )

        gate, lambda_map = (
            self.gate(
                feat,
                p_seg,
                s_poi,
            )
        )

        z = (
            torch.logit(
                p_seg.clamp(
                    EPS,
                    1.0
                    - EPS,
                )
            )
            + lambda_map
            * s_poi
        )

        p_final = torch.sigmoid(
            z
        )

        # Positive-only relative to pure SegFormer.
        if torch.any(
            p_final
            + 1e-7
            < p_seg
        ):
            raise RuntimeError(
                "Adaptive POI fusion decreased probability below SegFormer."
            )

        return {
            "gate": gate,
            "lambda_map": lambda_map,
            "p_final": p_final,
        }


# =============================================================================
# 9. LOSSES
# =============================================================================

def probability_loss(
    prob,
    target,
):
    probs = torch.cat(
        [
            1.0
            - prob,
            prob,
        ],
        dim=1,
    ).clamp(
        EPS,
        1.0,
    )

    return F.nll_loss(
        torch.log(
            probs
        ),
        target,
        ignore_index=IGNORE_VALUE,
    )


def balanced_gate_preference_loss(
    gate,
    s_poi,
    target,
):
    """
    For this positive-only POI formulation:
        GT school + POI support     -> prefer stronger lambda (gate -> 1)
        GT background + POI support -> prefer weaker lambda (gate -> 0)

    The two groups receive equal weight when both exist.
    """
    g = gate[
        :,
        0,
    ]

    s = s_poi[
        :,
        0,
    ]

    valid = (
        target
        != IGNORE_VALUE
    )

    informative = (
        s
        > GATE_INFORMATIVE_EPS
    )

    prefer_strong = (
        valid
        & informative
        & (
            target
            == 1
        )
    )

    prefer_weak = (
        valid
        & informative
        & (
            target
            == 0
        )
    )

    terms = []

    if torch.any(
        prefer_strong
    ):
        x = g[
            prefer_strong
        ].clamp(
            EPS,
            1.0
            - EPS,
        )

        terms.append(
            -torch.log(
                x
            ).mean()
        )

    if torch.any(
        prefer_weak
    ):
        x = g[
            prefer_weak
        ].clamp(
            EPS,
            1.0
            - EPS,
        )

        terms.append(
            -torch.log(
                1.0
                - x
            ).mean()
        )

    if not terms:
        loss = (
            gate.sum()
            * 0.0
        )

    else:
        loss = torch.stack(
            terms
        ).mean()

    return (
        loss,
        int(
            prefer_strong.sum().item()
        ),
        int(
            prefer_weak.sum().item()
        ),
    )


# =============================================================================
# 10. SLIDING-WINDOW LAMBDA INFERENCE
# =============================================================================

def get_positions(
    length,
    crop,
    stride,
):
    if length <= crop:
        return [
            0
        ]

    pos = list(
        range(
            0,
            length
            - crop
            + 1,
            stride,
        )
    )

    last = (
        length
        - crop
    )

    if pos[
        -1
    ] != last:
        pos.append(
            last
        )

    return pos


def pad_full_arrays(
    rgb,
    p_seg,
    s_poi,
):
    h, w = p_seg.shape

    pad_h = max(
        0,
        CROP_SIZE - h,
    )

    pad_w = max(
        0,
        CROP_SIZE - w,
    )

    if (
        pad_h == 0
        and pad_w == 0
    ):
        return (
            rgb,
            p_seg,
            s_poi,
            (
                h,
                w,
            ),
        )

    rgb_p = np.pad(
        rgb,
        (
            (
                0,
                pad_h,
            ),
            (
                0,
                pad_w,
            ),
            (
                0,
                0,
            ),
        ),
        mode="edge",
    )

    pseg_p = np.pad(
        p_seg,
        (
            (
                0,
                pad_h,
            ),
            (
                0,
                pad_w,
            ),
        ),
        mode="edge",
    )

    poi_p = np.pad(
        s_poi,
        (
            (
                0,
                pad_h,
            ),
            (
                0,
                pad_w,
            ),
        ),
        mode="constant",
        constant_values=0.0,
    )

    return (
        rgb_p,
        pseg_p,
        poi_p,
        (
            h,
            w,
        ),
    )


@torch.no_grad()
def predict_lambda_map_full(
    model,
    rgb,
    p_seg,
    s_poi,
    device,
):
    model.eval()

    (
        rgb_p,
        pseg_p,
        poi_p,
        original_hw,
    ) = pad_full_arrays(
        rgb,
        p_seg,
        s_poi,
    )

    h, w = pseg_p.shape

    ys = get_positions(
        h,
        CROP_SIZE,
        INFER_STRIDE,
    )

    xs = get_positions(
        w,
        CROP_SIZE,
        INFER_STRIDE,
    )

    lambda_sum = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.float32,
    )

    count = np.zeros(
        (
            h,
            w,
        ),
        dtype=np.float32,
    )

    for y0 in ys:
        for x0 in xs:
            rgb_tile = rgb_p[
                y0:
                y0
                + CROP_SIZE,
                x0:
                x0
                + CROP_SIZE,
            ]

            pseg_tile = pseg_p[
                y0:
                y0
                + CROP_SIZE,
                x0:
                x0
                + CROP_SIZE,
            ]

            poi_tile = poi_p[
                y0:
                y0
                + CROP_SIZE,
                x0:
                x0
                + CROP_SIZE,
            ]

            x = image_to_tensor(
                rgb_tile
            ).unsqueeze(
                0
            ).to(
                device
            )

            pseg_t = torch.from_numpy(
                pseg_tile
            ).float().unsqueeze(
                0
            ).unsqueeze(
                0
            ).to(
                device
            )

            poi_t = torch.from_numpy(
                poi_tile
            ).float().unsqueeze(
                0
            ).unsqueeze(
                0
            ).to(
                device
            )

            feat = model.decoder_feature(
                x
            )

            _, lambda_map = (
                model.gate(
                    feat,
                    pseg_t,
                    poi_t,
                )
            )

            lm = (
                lambda_map[
                    0,
                    0,
                ]
                .float()
                .cpu()
                .numpy()
            )

            lambda_sum[
                y0:
                y0
                + CROP_SIZE,
                x0:
                x0
                + CROP_SIZE,
            ] += lm

            count[
                y0:
                y0
                + CROP_SIZE,
                x0:
                x0
                + CROP_SIZE,
            ] += 1.0

    if np.any(
        count
        == 0
    ):
        raise RuntimeError(
            "Sliding-window lambda map has uncovered pixels."
        )

    lambda_map = (
        lambda_sum
        / count
    )

    oh, ow = original_hw

    return lambda_map[
        :oh,
        :ow,
    ].astype(
        np.float32
    )


def adaptive_probability_from_lambda(
    p_seg,
    s_poi,
    lambda_map,
):
    p = np_sigmoid(
        np_logit(
            p_seg
        )
        + lambda_map
        * s_poi
    )

    if np.any(
        p
        + 1e-7
        < p_seg
    ):
        raise RuntimeError(
            "Adaptive probability fell below SegFormer."
        )

    return p


# =============================================================================
# 11. FULL-SPLIT EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate_split(
    model,
    ids,
    seg_prob_dir,
    device,
    save_test=False,
):
    totals = {
        "SegFormer": [
            0,
            0,
            0,
            0,
        ],
        "Fixed SegFormer + POI": [
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
    }

    type_totals = {
        name: {
            method: [
                0,
                0,
                0,
                0,
            ]
            for method in totals
        }
        for name in SCHOOL_TYPES
    }

    type_meta = {
        name: {
            "positive_tiles": 0,
            "gt_target_pixels": 0,
        }
        for name in SCHOOL_TYPES
    }

    lambda_sum = 0.0
    lambda_n = 0
    lambda_poi_sum = 0.0
    lambda_poi_n = 0

    for i, gid in enumerate(
        ids,
        1,
    ):
        rgb = read_rgb(
            gid
        )

        gt = read_mask(
            gid
        )

        type_gt = read_type_mask(
            gid
        )

        p_seg = read_probability(
            seg_prob_dir,
            gid,
        )

        s_poi = empirical_school_prior(
            gid
        )

        if not (
            rgb.shape[
                :2
            ]
            == gt.shape
            == type_gt.shape
            == p_seg.shape
            == s_poi.shape
        ):
            raise RuntimeError(
                "Shape mismatch: {}".format(
                    gid
                )
            )

        p_fixed = fixed_poi_probability(
            p_seg,
            s_poi,
        )

        lambda_map = (
            predict_lambda_map_full(
                model=model,
                rgb=rgb,
                p_seg=p_seg,
                s_poi=s_poi,
                device=device,
            )
        )

        p_adapt = (
            adaptive_probability_from_lambda(
                p_seg,
                s_poi,
                lambda_map,
            )
        )

        probs = {
            "SegFormer": p_seg,
            "Fixed SegFormer + POI": p_fixed,
            "Adaptive visual-feature POI": p_adapt,
        }

        for method, prob in probs.items():
            add_counts(
                totals[
                    method
                ],
                counts_from_prob(
                    gt,
                    prob,
                ),
            )

        for name in SCHOOL_TYPES:
            target = (
                TYPE_TO_MASK_VALUE[
                    name
                ]
            )

            if not np.any(
                type_gt
                == target
            ):
                continue

            type_meta[
                name
            ][
                "positive_tiles"
            ] += 1

            type_meta[
                name
            ][
                "gt_target_pixels"
            ] += int(
                np.sum(
                    type_gt
                    == target
                )
            )

            for method, prob in probs.items():
                c = type_counts(
                    type_gt,
                    prob,
                    target,
                )

                add_counts(
                    type_totals[
                        name
                    ][
                        method
                    ],
                    c,
                )

        valid = (
            gt
            != IGNORE_VALUE
        )

        lambda_sum += float(
            np.sum(
                lambda_map[
                    valid
                ]
            )
        )

        lambda_n += int(
            np.sum(
                valid
            )
        )

        poi_mask = (
            valid
            & (
                s_poi
                > GATE_INFORMATIVE_EPS
            )
        )

        if np.any(
            poi_mask
        ):
            lambda_poi_sum += float(
                np.sum(
                    lambda_map[
                        poi_mask
                    ]
                )
            )

            lambda_poi_n += int(
                np.sum(
                    poi_mask
                )
            )

        if save_test:
            np.save(
                str(
                    TEST_ADAPT_PROB_DIR
                    / (
                        str(gid)
                        + "_school_probability.npy"
                    )
                ),
                p_adapt.astype(
                    np.float32
                ),
            )

            np.save(
                str(
                    TEST_LAMBDA_DIR
                    / (
                        str(gid)
                        + "_lambda_map.npy"
                    )
                ),
                lambda_map.astype(
                    np.float32
                ),
            )

            pred = (
                p_adapt
                >= THRESHOLD
            ).astype(
                np.uint8
            )

            Image.fromarray(
                pred
                * 255,
                mode="L",
            ).save(
                str(
                    TEST_BINARY_DIR
                    / (
                        str(gid)
                        + "_school.png"
                    )
                )
            )

        if (
            i % 10 == 0
            or i == len(
                ids
            )
        ):
            print(
                "  evaluate [{}/{}] {}"
                .format(
                    i,
                    len(
                        ids
                    ),
                    gid,
                )
            )

    metrics = {
        method: metrics_from_counts(
            c
        )
        for method, c in totals.items()
    }

    type_metrics = {}

    for name in SCHOOL_TYPES:
        type_metrics[
            name
        ] = {
            "label_cn": TYPE_TO_CN[
                name
            ],
            **type_meta[
                name
            ],
            "metrics": {
                method: metrics_from_counts(
                    c
                )
                for method, c
                in type_totals[
                    name
                ].items()
            },
        }

    diagnostics = {
        "mean_lambda_valid_pixels": (
            float(
                lambda_sum
                / max(
                    1,
                    lambda_n,
                )
            )
        ),
        "mean_lambda_where_poi_positive": (
            float(
                lambda_poi_sum
                / max(
                    1,
                    lambda_poi_n,
                )
            )
        ),
        "lambda_reference": float(
            LAMBDA_REF
        ),
        "lambda_max": float(
            2.0
            * LAMBDA_REF
        ),
    }

    return {
        "metrics": metrics,
        "type_metrics": type_metrics,
        "lambda_diagnostics": diagnostics,
    }


# =============================================================================
# 12. CHECKPOINT / SELECTION
# =============================================================================

def better_metric(
    new_m,
    old_m,
):
    if old_m is None:
        return True

    for key in [
        "iou",
        "f1",
        "precision",
        "recall",
    ]:
        a = float(
            new_m[
                key
            ]
        )

        b = float(
            old_m[
                key
            ]
        )

        if a > b + 1e-12:
            return True

        if a < b - 1e-12:
            return False

    return False


def save_checkpoint(
    path,
    gate,
    optimizer,
    epoch,
    best_val_metrics,
):
    torch.save(
        {
            "epoch": int(
                epoch
            ),
            "gate_state_dict": gate.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),
            "lambda_reference": float(
                LAMBDA_REF
            ),
            "lambda_max": float(
                2.0
                * LAMBDA_REF
            ),
            "best_val_metrics": best_val_metrics,
            "empirical_radii_m": {
                k: float(
                    v
                )
                for k, v
                in EMPIRICAL_RADIUS_M.items()
            },
        },
        str(
            path
        ),
    )


# =============================================================================
# 13. MAIN TRAIN / VAL / TEST
# =============================================================================

def main():
    ensure_dirs()
    check_inputs()
    set_seed(
        SEED
    )

    load_fixed_poi_config()

    train_ids = read_ids(
        FOLD_DIR
        / "train.txt"
    )

    val_ids = read_ids(
        FOLD_DIR
        / "val.txt"
    )

    test_ids = read_ids(
        FOLD_DIR
        / "test.txt"
    )

    if (
        len(
            train_ids
        ) != 295
        or len(
            val_ids
        ) != 42
        or len(
            test_ids
        ) != 84
    ):
        raise RuntimeError(
            "Expected Fold-0 split 295/42/84, got {}/{}/{}"
            .format(
                len(
                    train_ids
                ),
                len(
                    val_ids
                ),
                len(
                    test_ids
                ),
            )
        )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 100
    )

    print(
        "FOLD 0 | FIXED POI vs VISUAL-FEATURE ADAPTIVE POI GATE"
    )

    print(
        "=" * 100
    )

    print(
        "Primary comparison:"
    )

    print(
        "  Fixed SegFormer + POI  vs  Adaptive visual-feature SegFormer + POI"
    )

    print(
        "SegFormer itself is also reported only as context."
    )

    print(
        "Device:",
        device,
    )

    segformer = build_frozen_segformer(
        device
    )

    model = FrozenSegVisualLambdaNet(
        segformer=segformer,
        lambda_ref=LAMBDA_REF,
    ).to(
        device
    )

    # Only Gate parameters are optimized.
    optimizer = torch.optim.AdamW(
        model.gate.parameters(),
        lr=GATE_LR,
        weight_decay=WEIGHT_DECAY,
    )

    train_ds = GateCropDataset(
        train_ids
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    # -------------------------------------------------------------------------
    # Epoch 0 = exact current fixed SegFormer + POI reference.
    # -------------------------------------------------------------------------
    print()
    print(
        "=" * 100
    )
    print(
        "EPOCH 0 VALIDATION | MUST MATCH FIXED SEGFORMER + POI"
    )
    print(
        "=" * 100
    )

    val0 = evaluate_split(
        model=model,
        ids=val_ids,
        seg_prob_dir=VAL_SEG_PROB_DIR,
        device=device,
        save_test=False,
    )

    m_fixed_0 = val0[
        "metrics"
    ][
        "Fixed SegFormer + POI"
    ]

    m_adapt_0 = val0[
        "metrics"
    ][
        "Adaptive visual-feature POI"
    ]

    print(
        "Fixed   : P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(
            m_fixed_0[
                "precision"
            ],
            m_fixed_0[
                "recall"
            ],
            m_fixed_0[
                "f1"
            ],
            m_fixed_0[
                "iou"
            ],
        )
    )

    print(
        "Adaptive: P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(
            m_adapt_0[
                "precision"
            ],
            m_adapt_0[
                "recall"
            ],
            m_adapt_0[
                "f1"
            ],
            m_adapt_0[
                "iou"
            ],
        )
    )

    for key in [
        "precision",
        "recall",
        "f1",
        "iou",
        "accuracy",
    ]:
        if abs(
            float(
                m_fixed_0[
                    key
                ]
            )
            - float(
                m_adapt_0[
                    key
                ]
            )
        ) > 1e-7:
            raise RuntimeError(
                "Epoch-0 adaptive Gate does not exactly reproduce "
                "the fixed POI baseline for metric {}."
                .format(
                    key
                )
            )

    write_json(
        VAL_REFERENCE,
        {
            "epoch": 0,
            "fixed_metrics": m_fixed_0,
            "adaptive_metrics": m_adapt_0,
            "lambda_diagnostics": val0[
                "lambda_diagnostics"
            ],
            "integrity": (
                "epoch0 adaptive output equals current fixed SegFormer+POI"
            ),
        },
    )

    best_metrics = dict(
        m_adapt_0
    )

    best_epoch = 0

    save_checkpoint(
        BEST_CKPT,
        model.gate,
        optimizer=None,
        epoch=0,
        best_val_metrics=best_metrics,
    )

    log_rows = []

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    for epoch in range(
        1,
        EPOCHS
        + 1,
    ):
        model.train()

        sum_loss = 0.0
        sum_final = 0.0
        sum_pref = 0.0
        n_batches = 0

        prefer_strong_px = 0
        prefer_weak_px = 0

        for (
            x,
            y,
            pseg_t,
            poi_t,
        ) in train_loader:
            x = x.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            pseg_t = pseg_t.to(
                device,
                non_blocking=True,
            )

            poi_t = poi_t.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            out = model(
                pixel_values=x,
                p_seg=pseg_t,
                s_poi=poi_t,
            )

            loss_final = probability_loss(
                out[
                    "p_final"
                ],
                y,
            )

            (
                loss_pref,
                n_strong,
                n_weak,
            ) = balanced_gate_preference_loss(
                out[
                    "gate"
                ],
                poi_t,
                y,
            )

            loss = (
                loss_final
                + LAMBDA_GATE_PREF
                * loss_pref
            )

            loss.backward()

            optimizer.step()

            sum_loss += float(
                loss.item()
            )

            sum_final += float(
                loss_final.item()
            )

            sum_pref += float(
                loss_pref.item()
            )

            n_batches += 1

            prefer_strong_px += int(
                n_strong
            )

            prefer_weak_px += int(
                n_weak
            )

        row = {
            "epoch": int(
                epoch
            ),
            "train_loss": (
                sum_loss
                / max(
                    1,
                    n_batches,
                )
            ),
            "train_final_loss": (
                sum_final
                / max(
                    1,
                    n_batches,
                )
            ),
            "train_gate_pref_loss": (
                sum_pref
                / max(
                    1,
                    n_batches,
                )
            ),
            "prefer_stronger_poi_pixels": int(
                prefer_strong_px
            ),
            "prefer_weaker_poi_pixels": int(
                prefer_weak_px
            ),
        }

        do_val = (
            epoch
            % VAL_EVERY
            == 0
            or epoch
            == EPOCHS
        )

        if do_val:
            print()
            print(
                "=" * 100
            )

            print(
                "VALIDATION EPOCH {}"
                .format(
                    epoch
                )
            )

            print(
                "=" * 100
            )

            val = evaluate_split(
                model=model,
                ids=val_ids,
                seg_prob_dir=VAL_SEG_PROB_DIR,
                device=device,
                save_test=False,
            )

            vm = val[
                "metrics"
            ][
                "Adaptive visual-feature POI"
            ]

            row.update({
                "val_precision": vm[
                    "precision"
                ],
                "val_recall": vm[
                    "recall"
                ],
                "val_f1": vm[
                    "f1"
                ],
                "val_iou": vm[
                    "iou"
                ],
                "val_mean_lambda_poi": (
                    val[
                        "lambda_diagnostics"
                    ][
                        "mean_lambda_where_poi_positive"
                    ]
                ),
            })

            print(
                "Adaptive | P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f} | "
                "mean lambda on POI support={:.4f}"
                .format(
                    vm[
                        "precision"
                    ],
                    vm[
                        "recall"
                    ],
                    vm[
                        "f1"
                    ],
                    vm[
                        "iou"
                    ],
                    val[
                        "lambda_diagnostics"
                    ][
                        "mean_lambda_where_poi_positive"
                    ],
                )
            )

            if better_metric(
                vm,
                best_metrics,
            ):
                best_metrics = dict(
                    vm
                )

                best_epoch = int(
                    epoch
                )

                save_checkpoint(
                    BEST_CKPT,
                    model.gate,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_val_metrics=best_metrics,
                )

                print(
                    "  -> new best checkpoint"
                )

        log_rows.append(
            row
        )

        pd.DataFrame(
            log_rows
        ).to_csv(
            TRAIN_LOG,
            index=False,
            encoding="utf-8-sig",
        )

        save_checkpoint(
            LAST_CKPT,
            model.gate,
            optimizer=optimizer,
            epoch=epoch,
            best_val_metrics=best_metrics,
        )

        print(
            "Epoch {:02d}/{:02d} | loss={:.6f} final={:.6f} pref={:.6f}"
            .format(
                epoch,
                EPOCHS,
                row[
                    "train_loss"
                ],
                row[
                    "train_final_loss"
                ],
                row[
                    "train_gate_pref_loss"
                ],
            )
        )

    # -------------------------------------------------------------------------
    # Load selected Gate
    # -------------------------------------------------------------------------
    best_obj = torch.load(
        str(
            BEST_CKPT
        ),
        map_location=device,
    )

    model.gate.load_state_dict(
        best_obj[
            "gate_state_dict"
        ],
        strict=True,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "SELECTED GATE"
    )

    print(
        "=" * 100
    )

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation IoU:",
        best_metrics[
            "iou"
        ],
    )

    selected = {
        "fold_id": 0,
        "diagnostic_only": True,
        "primary_comparison": (
            "fixed SegFormer+POI vs visual-feature adaptive SegFormer+POI"
        ),
        "segformer_frozen": True,
        "poi_radii_fixed": True,
        "empirical_radius_m": {
            k: float(
                v
            )
            for k, v
            in EMPIRICAL_RADIUS_M.items()
        },
        "lambda_reference": float(
            LAMBDA_REF
        ),
        "lambda_map_formula": (
            "lambda(x)=2*lambda_ref*sigmoid(Gate("
            "SegFormer_decoder_feature,P_seg,S_poi))"
        ),
        "lambda_range": [
            0.0,
            float(
                2.0
                * LAMBDA_REF
            ),
        ],
        "epoch0_exact_fixed_baseline": True,
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_metrics": best_metrics,
        "training": {
            "epochs": int(
                EPOCHS
            ),
            "val_every": int(
                VAL_EVERY
            ),
            "patches_per_image_per_epoch": int(
                PATCHES_PER_IMAGE_PER_EPOCH
            ),
            "gate_lr": float(
                GATE_LR
            ),
            "gate_pref_loss_weight": float(
                LAMBDA_GATE_PREF
            ),
        },
    }

    write_json(
        SELECTED_CONFIG,
        selected,
    )

    # -------------------------------------------------------------------------
    # TEST ONCE
    # -------------------------------------------------------------------------
    print()
    print(
        "=" * 100
    )

    print(
        "HELD-OUT TEST: ONCE"
    )

    print(
        "=" * 100
    )

    test = evaluate_split(
        model=model,
        ids=test_ids,
        seg_prob_dir=TEST_SEG_PROB_DIR,
        device=device,
        save_test=True,
    )

    tm = test[
        "metrics"
    ]

    print()
    print(
        "{:<34s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"
        .format(
            "METHOD",
            "P",
            "R",
            "F1",
            "IoU",
            "ACC",
        )
    )

    for method in [
        "SegFormer",
        "Fixed SegFormer + POI",
        "Adaptive visual-feature POI",
    ]:
        m = tm[
            method
        ]

        print(
            "{:<34s} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}"
            .format(
                method,
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

    fixed = tm[
        "Fixed SegFormer + POI"
    ]

    adapt = tm[
        "Adaptive visual-feature POI"
    ]

    print()
    print(
        "DIRECT DELTA: adaptive minus fixed SegFormer+POI"
    )

    print(
        "  dP={:+.4f} dR={:+.4f} dF1={:+.4f} dIoU={:+.4f}"
        .format(
            adapt[
                "precision"
            ]
            - fixed[
                "precision"
            ],
            adapt[
                "recall"
            ]
            - fixed[
                "recall"
            ],
            adapt[
                "f1"
            ]
            - fixed[
                "f1"
            ],
            adapt[
                "iou"
            ]
            - fixed[
                "iou"
            ],
        )
    )

    print()
    print(
        "Lambda diagnostics:"
    )

    print(
        "  reference lambda              = {:.4f}"
        .format(
            LAMBDA_REF
        )
    )

    print(
        "  mean lambda on valid pixels   = {:.4f}"
        .format(
            test[
                "lambda_diagnostics"
            ][
                "mean_lambda_valid_pixels"
            ]
        )
    )

    print(
        "  mean lambda where POI > 0     = {:.4f}"
        .format(
            test[
                "lambda_diagnostics"
            ][
                "mean_lambda_where_poi_positive"
            ]
        )
    )

    # Type-specific output
    type_rows = []

    print()
    print(
        "TEST BY SCHOOL TYPE"
    )

    for name in SCHOOL_TYPES:
        x = test[
            "type_metrics"
        ][
            name
        ]

        fm = x[
            "metrics"
        ][
            "Fixed SegFormer + POI"
        ]

        am = x[
            "metrics"
        ][
            "Adaptive visual-feature POI"
        ]

        print(
            "  {} | tiles={} GT={} | "
            "P {:.4f}->{:.4f} ({:+.4f}) | "
            "R {:.4f}->{:.4f} ({:+.4f}) | "
            "IoU {:.4f}->{:.4f} ({:+.4f})"
            .format(
                TYPE_TO_CN[
                    name
                ],
                x[
                    "positive_tiles"
                ],
                x[
                    "gt_target_pixels"
                ],
                fm[
                    "precision"
                ],
                am[
                    "precision"
                ],
                am[
                    "precision"
                ]
                - fm[
                    "precision"
                ],
                fm[
                    "recall"
                ],
                am[
                    "recall"
                ],
                am[
                    "recall"
                ]
                - fm[
                    "recall"
                ],
                fm[
                    "iou"
                ],
                am[
                    "iou"
                ],
                am[
                    "iou"
                ]
                - fm[
                    "iou"
                ],
            )
        )

        type_rows.append({
            "school_type": name,
            "school_type_cn": TYPE_TO_CN[
                name
            ],
            "positive_tiles": x[
                "positive_tiles"
            ],
            "gt_target_pixels": x[
                "gt_target_pixels"
            ],
            "fixed_precision": fm[
                "precision"
            ],
            "fixed_recall": fm[
                "recall"
            ],
            "fixed_f1": fm[
                "f1"
            ],
            "fixed_iou": fm[
                "iou"
            ],
            "adaptive_precision": am[
                "precision"
            ],
            "adaptive_recall": am[
                "recall"
            ],
            "adaptive_f1": am[
                "f1"
            ],
            "adaptive_iou": am[
                "iou"
            ],
            "delta_precision": (
                am[
                    "precision"
                ]
                - fm[
                    "precision"
                ]
            ),
            "delta_recall": (
                am[
                    "recall"
                ]
                - fm[
                    "recall"
                ]
            ),
            "delta_f1": (
                am[
                    "f1"
                ]
                - fm[
                    "f1"
                ]
            ),
            "delta_iou": (
                am[
                    "iou"
                ]
                - fm[
                    "iou"
                ]
            ),
        })

    pd.DataFrame(
        type_rows
    ).to_csv(
        TEST_TYPE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    compare_rows = []

    for method in [
        "SegFormer",
        "Fixed SegFormer + POI",
        "Adaptive visual-feature POI",
    ]:
        compare_rows.append({
            "method": method,
            **tm[
                method
            ],
        })

    compare_rows.append({
        "method": (
            "Delta adaptive minus fixed SegFormer+POI"
        ),
        "precision": (
            adapt[
                "precision"
            ]
            - fixed[
                "precision"
            ]
        ),
        "recall": (
            adapt[
                "recall"
            ]
            - fixed[
                "recall"
            ]
        ),
        "f1": (
            adapt[
                "f1"
            ]
            - fixed[
                "f1"
            ]
        ),
        "iou": (
            adapt[
                "iou"
            ]
            - fixed[
                "iou"
            ]
        ),
        "accuracy": (
            adapt[
                "accuracy"
            ]
            - fixed[
                "accuracy"
            ]
        ),
        "TP": (
            adapt[
                "TP"
            ]
            - fixed[
                "TP"
            ]
        ),
        "TN": (
            adapt[
                "TN"
            ]
            - fixed[
                "TN"
            ]
        ),
        "FP": (
            adapt[
                "FP"
            ]
            - fixed[
                "FP"
            ]
        ),
        "FN": (
            adapt[
                "FN"
            ]
            - fixed[
                "FN"
            ]
        ),
    })

    pd.DataFrame(
        compare_rows
    ).to_csv(
        TEST_COMPARE_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    write_json(
        TEST_SUMMARY,
        {
            "selected_config": selected,
            "test_metrics": tm,
            "type_metrics": test[
                "type_metrics"
            ],
            "lambda_diagnostics": test[
                "lambda_diagnostics"
            ],
            "direct_delta_adaptive_minus_fixed": {
                key: float(
                    adapt[
                        key
                    ]
                    - fixed[
                        key
                    ]
                )
                for key in [
                    "precision",
                    "recall",
                    "f1",
                    "iou",
                    "accuracy",
                ]
            },
        },
    )

    print()
    print(
        "Output:"
    )

    print(
        " ",
        OUT_ROOT,
    )


if __name__ == "__main__":
    main()
