# -*- coding: utf-8 -*-
r"""
SegFormer-B2 pure-visual binary baseline for the new school 5-fold dataset
==========================================================================

Adapted directly from the uploaded:
    44b_segformer_b2_pure_visual_core01_fix.py

Training task remains STRICTLY BINARY:
    0   = background inside the true core
    1   = school-building pixel inside the true core
    255 = outside the true core (context only / ignored)

No POI, AOI, school type, or VLM information is used by the model.

Prepared dataset:
    images/
    masks_png/          -> 0 / 1 / 255, USED for training/evaluation
    masks_type_png/     -> 0 / 1 / 2 / 3 / 255, EVALUATION ONLY
    folds/fold_0 ... fold_4

Type-mask semantics:
    0   = background
    1   = primary-school building
    2   = secondary-school building
    3   = higher-education building
    255 = outside true core

Type-specific metrics:
- model prediction is still only school/background;
- for one target type, only tiles containing that type contribute;
- target type = positive;
- background = negative;
- the other two school-building types = ignored.

Therefore the type-specific Precision / Recall / F1 / IoU values are
statistics of the SAME binary prediction, not a 3-class model.

Fold control:
    RUN_FOLDS = [0]                  # first experiment
    RUN_FOLDS = [0, 1, 2, 3, 4]    # complete experiment

If SKIP_COMPLETED_FOLDS=True, a fold with an existing final
test_summary.json is not rerun. Thus Fold 0 can be reused later.

The following training settings are intentionally unchanged from the
uploaded baseline:
- SegFormer-B2 ADE20K-pretrained checkpoint
- ImageNet normalization
- AdamW, lr=6e-5, weight_decay=0.01
- polynomial LR decay
- 512x512 crop
- 2 random crops per training image per epoch
- school-center probability 0.65
- geometric augmentation
- 30 epochs
- validation every 5 epochs
- 512/256 sliding-window inference
- CrossEntropyLoss(ignore_index=255)
- best checkpoint selected by OVERALL validation micro IoU only
- held-out test evaluated using that best checkpoint
"""

import os
import csv
import json
import math
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError:
    raise RuntimeError(
        "Missing transformers.\n"
        "Install it in the current environment with:\n"
        "    pip install transformers"
    )


# ============================================================
# 1. PATHS
# ============================================================

DATASET_ROOT = Path(__file__).resolve().parent / "data"

VERSION_FILE = DATASET_ROOT / "dataset_version.txt"

if not VERSION_FILE.exists():
    raise FileNotFoundError(
        "Prepared dataset version file not found:\n{}\n"
        "Run prepare_school_segformer_5fold_overlap_safe_fix2.py first."
        .format(VERSION_FILE)
    )

DATASET_VERSION = VERSION_FILE.read_text(
    encoding="utf-8"
).strip()

if not DATASET_VERSION:
    raise RuntimeError(
        "dataset_version.txt is empty."
    )

IMAGE_DIR = DATASET_ROOT / "images"
MASK_DIR = DATASET_ROOT / "masks_png"
TYPE_MASK_DIR = DATASET_ROOT / "masks_type_png"
FOLDS_DIR = DATASET_ROOT / "folds"

OUT_BASE_ROOT = Path(__file__).resolve().parent / "outputs" / "segformer_5fold"

OUT_BASE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)

PRETRAINED_MODEL = (
    Path(__file__).resolve().parent / "pretrained" / "segformer_b2"
)

# Fold-specific paths; configured by configure_fold_paths().
OUT_ROOT = None
CKPT_DIR = None
VAL_BEST_DIR = None
TEST_OUT_DIR = None
BEST_CKPT = None
LAST_CKPT = None
TRAIN_LOG = None
MASK_AUDIT_CSV = None


# ============================================================
# 2. SETTINGS
# ============================================================

SEED = 42

# First experiment:
RUN_FOLDS = [0]

# Full experiment later:
# RUN_FOLDS = [0, 1, 2, 3, 4]

SKIP_COMPLETED_FOLDS = True

CROP_SIZE = 512
INFER_STRIDE = 256

BATCH_SIZE = 1
EPOCHS = 30
PATCHES_PER_IMAGE_PER_EPOCH = 2
VAL_EVERY = 5

BASE_LR = 6e-5
WEIGHT_DECAY = 0.01
POLY_POWER = 0.9

IGNORE_VALUE = 255

SCHOOL_CENTER_PROB = 0.65
MIN_VALID_PIXELS_PER_CROP = 1024
MAX_CROP_TRIES = 30
USE_AUGMENTATION = True
IMAGE_CACHE_SIZE = 12

RESUME_IF_AVAILABLE = True

OVERLAY_ALPHA = 0.45
SCHOOL_COLOR = np.array(
    [0, 255, 0],
    dtype=np.uint8
)

TYPE_VALUE_TO_KEY = {
    1: "primary",
    2: "secondary",
    3: "higher_education",
}

TYPE_KEY_TO_LABEL = {
    "primary": "小学",
    "secondary": "中学",
    "higher_education": "高等教育",
}

IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
).reshape(1, 1, 3)


# ============================================================
# 3. HELPERS
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_fold_paths(
    fold_id
):
    global OUT_ROOT
    global CKPT_DIR
    global VAL_BEST_DIR
    global TEST_OUT_DIR
    global BEST_CKPT
    global LAST_CKPT
    global TRAIN_LOG
    global MASK_AUDIT_CSV

    OUT_ROOT = (
        OUT_BASE_ROOT
        / "fold_{}".format(
            int(fold_id)
        )
    )

    CKPT_DIR = OUT_ROOT / "checkpoints"
    VAL_BEST_DIR = OUT_ROOT / "val_predictions_best"
    TEST_OUT_DIR = OUT_ROOT / "test_predictions"

    for d in [
        OUT_ROOT,
        CKPT_DIR,
        VAL_BEST_DIR,
        TEST_OUT_DIR,
    ]:
        d.mkdir(
            parents=True,
            exist_ok=True
        )

    BEST_CKPT = CKPT_DIR / "best_model.pth"
    LAST_CKPT = CKPT_DIR / "last_checkpoint.pth"
    TRAIN_LOG = OUT_ROOT / "training_log.csv"
    MASK_AUDIT_CSV = OUT_ROOT / "prepared_mask_audit.csv"


def get_fold_paths(
    fold_id
):
    split_dir = (
        FOLDS_DIR
        / "fold_{}".format(
            int(fold_id)
        )
    )

    return {
        "split_dir": split_dir,
        "train": split_dir / "train.txt",
        "val": split_dir / "val.txt",
        "test": split_dir / "test.txt",
    }


def check_inputs():
    required = [
        DATASET_ROOT,
        IMAGE_DIR,
        MASK_DIR,
        TYPE_MASK_DIR,
        FOLDS_DIR,
        PRETRAINED_MODEL,
    ]

    for p in required:
        p = Path(p)

        if not p.exists():
            raise FileNotFoundError(
                "Required path not found:\n{}".format(
                    p
                )
            )

    if not RUN_FOLDS:
        raise RuntimeError(
            "RUN_FOLDS is empty."
        )

    for fold_id in RUN_FOLDS:

        fold_id = int(
            fold_id
        )

        if fold_id not in {
            0, 1, 2, 3, 4
        }:
            raise ValueError(
                "Invalid fold_id in RUN_FOLDS: {}".format(
                    fold_id
                )
            )

        paths = get_fold_paths(
            fold_id
        )

        for key in [
            "train",
            "val",
            "test",
        ]:
            if not paths[key].exists():
                raise FileNotFoundError(
                    "Fold {} {} list not found:\n{}".format(
                        fold_id,
                        key,
                        paths[key],
                    )
                )


def read_id_list(path):
    ids = []
    with open(str(path), "r", encoding="utf-8") as f:
        for line in f:
            x = line.strip()
            if x:
                ids.append(x)

    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate grid IDs in {}".format(path)
        )
    return ids


def find_file(folder, grid_id, suffixes):
    for suffix in suffixes:
        p = folder / "{}{}".format(grid_id, suffix)
        if p.exists():
            return p

    allowed = set(x.lower() for x in suffixes)
    matches = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.stem == grid_id
        and p.suffix.lower() in allowed
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            "Cannot uniquely locate {} in {}\nmatches={}".format(
                grid_id, folder, matches
            )
        )
    return matches[0]


# ---------------------------------------------------------------------------
# PREPARED MASKS
# ---------------------------------------------------------------------------

def read_raw_mask(path):
    """
    Prepared binary mask:
        0   background inside true core
        1   school building inside true core
        255 outside true core
    """
    with Image.open(
        str(path)
    ) as im:

        arr = np.asarray(
            im.convert("L"),
            dtype=np.uint8
        )

    values = set(
        np.unique(
            arr
        ).tolist()
    )

    if not values.issubset(
        {
            0,
            1,
            255,
        }
    ):
        raise RuntimeError(
            "Unexpected prepared binary-mask values in {}: {}"
            .format(
                path,
                sorted(values)
            )
        )

    return arr


def read_mask(
    path,
    gid=None
):
    """
    masks_png already contains the FINAL 0/1/255 target.
    No core reconstruction is performed during training.
    """
    return read_raw_mask(
        path
    )


def read_type_mask(path):
    """
    Evaluation-only mask:
        0   background
        1   primary school
        2   secondary school
        3   higher education
        255 outside true core
    """
    with Image.open(
        str(path)
    ) as im:

        arr = np.asarray(
            im.convert("L"),
            dtype=np.uint8
        )

    values = set(
        np.unique(
            arr
        ).tolist()
    )

    if not values.issubset(
        {
            0,
            1,
            2,
            3,
            255,
        }
    ):
        raise RuntimeError(
            "Unexpected type-mask values in {}: {}"
            .format(
                path,
                sorted(values)
            )
        )

    return arr


def write_prepared_mask_audit(
    grid_ids
):
    """
    Verify:
        binary 1   <=> type 1/2/3
        binary 0   <=> type 0
        binary 255 <=> type 255
    """
    rows = []

    totals = {
        "primary": 0,
        "secondary": 0,
        "higher": 0,
        "school": 0,
        "background": 0,
        "ignore": 0,
    }

    for gid in grid_ids:

        binary_path = find_file(
            MASK_DIR,
            gid,
            [".png"]
        )

        type_path = find_file(
            TYPE_MASK_DIR,
            gid,
            [".png"]
        )

        binary = read_mask(
            binary_path,
            gid
        )

        type_mask = read_type_mask(
            type_path
        )

        if binary.shape != type_mask.shape:
            raise RuntimeError(
                "Binary/type mask shape mismatch for {}: {} vs {}"
                .format(
                    gid,
                    binary.shape,
                    type_mask.shape,
                )
            )

        expected = np.full(
            type_mask.shape,
            IGNORE_VALUE,
            dtype=np.uint8,
        )

        valid = (
            type_mask
            != IGNORE_VALUE
        )

        expected[
            valid
        ] = np.where(
            type_mask[
                valid
            ] > 0,
            1,
            0,
        ).astype(
            np.uint8
        )

        mismatch = int(
            np.sum(
                binary != expected
            )
        )

        if mismatch > 0:
            raise RuntimeError(
                "{}: binary/type masks inconsistent at {} pixels."
                .format(
                    gid,
                    mismatch
                )
            )

        n_primary = int(
            np.sum(
                type_mask == 1
            )
        )
        n_secondary = int(
            np.sum(
                type_mask == 2
            )
        )
        n_higher = int(
            np.sum(
                type_mask == 3
            )
        )
        n_school = int(
            np.sum(
                binary == 1
            )
        )
        n_background = int(
            np.sum(
                binary == 0
            )
        )
        n_ignore = int(
            np.sum(
                binary == IGNORE_VALUE
            )
        )

        totals["primary"] += n_primary
        totals["secondary"] += n_secondary
        totals["higher"] += n_higher
        totals["school"] += n_school
        totals["background"] += n_background
        totals["ignore"] += n_ignore

        rows.append({
            "grid_id": gid,
            "primary_pixels": n_primary,
            "secondary_pixels": n_secondary,
            "higher_education_pixels": n_higher,
            "school_pixels_binary": n_school,
            "background_pixels": n_background,
            "ignore_pixels": n_ignore,
            "binary_type_mismatch_pixels": mismatch,
        })

    write_rows_csv(
        MASK_AUDIT_CSV,
        rows
    )

    print()
    print("Prepared-mask audit:")
    print(
        "  primary pixels         : {}".format(
            totals["primary"]
        )
    )
    print(
        "  secondary pixels       : {}".format(
            totals["secondary"]
        )
    )
    print(
        "  higher-education pixels: {}".format(
            totals["higher"]
        )
    )
    print(
        "  all school pixels      : {}".format(
            totals["school"]
        )
    )
    print(
        "  core background pixels : {}".format(
            totals["background"]
        )
    )
    print(
        "  outside-core ignore    : {}".format(
            totals["ignore"]
        )
    )
    print(
        "  audit CSV              : {}".format(
            MASK_AUDIT_CSV
        )
    )


def read_rgb(path):
    with Image.open(str(path)) as im:
        arr = np.asarray(
            im.convert("RGB"),
            dtype=np.uint8
        )
    return arr


def image_to_tensor(rgb):
    """
    uint8 RGB HWC -> float CHW using the standard ImageNet normalization
    expected by SegFormer pretrained checkpoints.
    """
    x = rgb.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(
        np.ascontiguousarray(x.transpose(2, 0, 1))
    ).float()
    return x


# ============================================================
# 4. TRAIN DATASET
# ============================================================

class SchoolCropDataset(Dataset):

    def __init__(
        self,
        grid_ids,
        image_dir,
        mask_dir,
        crop_size,
        patches_per_image,
        school_center_prob,
        min_valid_pixels,
        max_tries,
        augment,
        cache_size,
    ):
        self.grid_ids = list(grid_ids)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.crop_size = int(crop_size)
        self.patches_per_image = int(patches_per_image)
        self.school_center_prob = float(school_center_prob)
        self.min_valid_pixels = int(min_valid_pixels)
        self.max_tries = int(max_tries)
        self.augment = bool(augment)
        self.cache_size = int(cache_size)

        self.image_paths = {}
        self.mask_paths = {}
        self.school_coords = {}
        self.valid_coords = {}
        self.cache = OrderedDict()

        print("Preparing training crop sampler...")

        for i, gid in enumerate(self.grid_ids, 1):
            image_path = find_file(
                self.image_dir,
                gid,
                [".tif", ".tiff"]
            )
            mask_path = find_file(
                self.mask_dir,
                gid,
                [".png"]
            )

            self.image_paths[gid] = image_path
            self.mask_paths[gid] = mask_path

            mask = read_mask(mask_path, gid)

            sy, sx = np.where(mask == 1)
            school = (
                np.stack([sy, sx], axis=1)
                if len(sy) > 0
                else np.empty((0, 2), dtype=np.int32)
            )

            vy, vx = np.where(mask != IGNORE_VALUE)
            valid = (
                np.stack([vy, vx], axis=1)
                if len(vy) > 0
                else np.empty((0, 2), dtype=np.int32)
            )

            if valid.shape[0] == 0:
                raise RuntimeError(
                    "Training tile {} has zero supervised pixels.".format(gid)
                )

            school = self._cap_coords(
                school, 30000, gid + "_school"
            )
            valid = self._cap_coords(
                valid, 50000, gid + "_valid"
            )

            self.school_coords[gid] = school
            self.valid_coords[gid] = valid

            if i % 10 == 0 or i == len(self.grid_ids):
                print(
                    "  [{}/{}] {}".format(
                        i, len(self.grid_ids), gid
                    )
                )

        print("Training crop sampler ready.")
        print()

    @staticmethod
    def _cap_coords(coords, max_n, seed_string):
        if coords.shape[0] <= max_n:
            return coords.astype(np.int32)

        seed = (
            sum(ord(ch) for ch in seed_string)
            + SEED
        ) % (2 ** 31 - 1)

        rng = np.random.RandomState(seed)
        idx = rng.choice(
            coords.shape[0],
            size=max_n,
            replace=False
        )
        return coords[idx].astype(np.int32)

    def __len__(self):
        return len(self.grid_ids) * self.patches_per_image

    def _load_pair(self, gid):
        if gid in self.cache:
            rgb, mask = self.cache.pop(gid)
            self.cache[gid] = (rgb, mask)
            return rgb, mask

        rgb = read_rgb(self.image_paths[gid])
        mask = read_mask(self.mask_paths[gid], gid)

        if rgb.shape[:2] != mask.shape:
            raise RuntimeError(
                "Image/mask shape mismatch for {}: {} vs {}".format(
                    gid, rgb.shape[:2], mask.shape
                )
            )

        self.cache[gid] = (rgb, mask)

        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

        return rgb, mask

    def _choose_center(self, gid):
        school = self.school_coords[gid]
        valid = self.valid_coords[gid]

        use_school = (
            school.shape[0] > 0
            and random.random() < self.school_center_prob
        )

        pool = school if use_school else valid

        j = random.randrange(pool.shape[0])
        return int(pool[j, 0]), int(pool[j, 1])

    def _crop_bounds(self, h, w, cy, cx):
        size = self.crop_size

        if h <= size:
            y0 = 0
        else:
            y0 = max(
                0,
                min(
                    cy - size // 2,
                    h - size
                )
            )

        if w <= size:
            x0 = 0
        else:
            x0 = max(
                0,
                min(
                    cx - size // 2,
                    w - size
                )
            )

        return y0, x0

    def _pad_if_needed(self, rgb, mask):
        h, w = mask.shape

        ph = max(0, self.crop_size - h)
        pw = max(0, self.crop_size - w)

        if ph == 0 and pw == 0:
            return rgb, mask

        mode = (
            "reflect"
            if h > 1 and w > 1
            else "edge"
        )

        rgb = np.pad(
            rgb,
            ((0, ph), (0, pw), (0, 0)),
            mode=mode
        )

        mask = np.pad(
            mask,
            ((0, ph), (0, pw)),
            mode="constant",
            constant_values=IGNORE_VALUE
        )

        return rgb, mask

    def _augment(self, rgb, mask):
        if not self.augment:
            return rgb, mask

        if random.random() < 0.5:
            rgb = np.flip(rgb, axis=1)
            mask = np.flip(mask, axis=1)

        if random.random() < 0.5:
            rgb = np.flip(rgb, axis=0)
            mask = np.flip(mask, axis=0)

        k = random.randint(0, 3)
        if k:
            rgb = np.rot90(rgb, k=k, axes=(0, 1))
            mask = np.rot90(mask, k=k, axes=(0, 1))

        return (
            np.ascontiguousarray(rgb),
            np.ascontiguousarray(mask)
        )

    def __getitem__(self, index):
        gid = self.grid_ids[
            index % len(self.grid_ids)
        ]

        rgb, mask = self._load_pair(gid)
        rgb, mask = self._pad_if_needed(
            rgb, mask
        )

        h, w = mask.shape

        best = None
        best_valid = -1

        for _ in range(self.max_tries):
            cy, cx = self._choose_center(gid)
            y0, x0 = self._crop_bounds(
                h, w, cy, cx
            )

            crop_rgb = rgb[
                y0:y0 + self.crop_size,
                x0:x0 + self.crop_size
            ]
            crop_mask = mask[
                y0:y0 + self.crop_size,
                x0:x0 + self.crop_size
            ]

            valid_n = int(
                np.sum(
                    crop_mask != IGNORE_VALUE
                )
            )

            if valid_n > best_valid:
                best = (crop_rgb, crop_mask)
                best_valid = valid_n

            if valid_n >= self.min_valid_pixels:
                break

        crop_rgb, crop_mask = best
        crop_rgb, crop_mask = self._augment(
            crop_rgb, crop_mask
        )

        x = image_to_tensor(crop_rgb)
        y = torch.from_numpy(
            crop_mask.astype(np.int64)
        ).long()

        return x, y, gid


# ============================================================
# 5. MODEL
# ============================================================

def build_model(device):
    print(
        "Loading SegFormer-B2 pretrained model:\n  {}".format(
            PRETRAINED_MODEL
        )
    )

    # The pretrained semantic classifier has ADE20K classes.
    # ignore_mismatched_sizes=True keeps all compatible pretrained weights
    # while replacing the final classifier for the binary school task.
    model = SegformerForSemanticSegmentation.from_pretrained(
        PRETRAINED_MODEL,
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

    model = model.to(device)

    print("SegFormer-B2 created.")
    print("Binary output order: 0=background, 1=school")

    return model


def make_optimizer(model):
    return torch.optim.AdamW(
        [
            p for p in model.parameters()
            if p.requires_grad
        ],
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )


def update_poly_lr(
    optimizer,
    epoch_index
):
    progress = float(
        epoch_index
    ) / float(
        max(1, EPOCHS)
    )

    factor = pow(
        max(0.0, 1.0 - progress),
        POLY_POWER
    )

    lr = BASE_LR * factor

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


def forward_logits(
    model,
    x,
    out_hw
):
    outputs = model(
        pixel_values=x
    )

    logits = outputs.logits

    if logits.shape[-2:] != out_hw:
        logits = F.interpolate(
            logits,
            size=out_hw,
            mode="bilinear",
            align_corners=False,
        )

    return logits


# ============================================================
# 6. FULL-IMAGE INFERENCE
# ============================================================

def get_positions(
    length,
    tile_size,
    stride
):
    if length <= tile_size:
        return [0]

    pos = list(
        range(
            0,
            length - tile_size + 1,
            stride
        )
    )

    last = length - tile_size

    if pos[-1] != last:
        pos.append(last)

    return pos


def pad_small_image(rgb):
    h, w, _ = rgb.shape

    ph = max(
        0,
        CROP_SIZE - h
    )
    pw = max(
        0,
        CROP_SIZE - w
    )

    if ph == 0 and pw == 0:
        return rgb, (h, w)

    mode = (
        "reflect"
        if h > 1 and w > 1
        else "edge"
    )

    rgb = np.pad(
        rgb,
        ((0, ph), (0, pw), (0, 0)),
        mode=mode
    )

    return rgb, (h, w)


@torch.no_grad()
def predict_full_image(
    model,
    rgb,
    device,
    print_windows=False
):
    model.eval()

    padded, original_hw = (
        pad_small_image(rgb)
    )

    h, w = padded.shape[:2]

    ys = get_positions(
        h,
        CROP_SIZE,
        INFER_STRIDE
    )
    xs = get_positions(
        w,
        CROP_SIZE,
        INFER_STRIDE
    )

    prob_sum = np.zeros(
        (2, h, w),
        dtype=np.float32
    )
    count = np.zeros(
        (h, w),
        dtype=np.float32
    )

    total = len(ys) * len(xs)
    done = 0

    for y0 in ys:
        for x0 in xs:
            tile = padded[
                y0:y0 + CROP_SIZE,
                x0:x0 + CROP_SIZE
            ]

            x = (
                image_to_tensor(tile)
                .unsqueeze(0)
                .to(
                    device,
                    non_blocking=True
                )
            )

            logits = forward_logits(
                model,
                x,
                (CROP_SIZE, CROP_SIZE)
            )

            probs = (
                torch.softmax(
                    logits,
                    dim=1
                )[0]
                .float()
                .cpu()
                .numpy()
            )

            prob_sum[
                :,
                y0:y0 + CROP_SIZE,
                x0:x0 + CROP_SIZE
            ] += probs

            count[
                y0:y0 + CROP_SIZE,
                x0:x0 + CROP_SIZE
            ] += 1.0

            done += 1

            if print_windows:
                print(
                    "\r    window {}/{}".format(
                        done, total
                    ),
                    end="",
                    flush=True
                )

    if print_windows:
        print()

    if np.any(count == 0):
        raise RuntimeError(
            "Some pixels were not covered by inference windows."
        )

    prob = (
        prob_sum
        / count[None, :, :]
    )

    pred_school = np.argmax(
        prob,
        axis=0
    ).astype(np.uint8)

    # Class 1 = school
    pred_school = (
        pred_school == 1
    ).astype(np.uint8)

    school_prob = prob[1]

    oh, ow = original_hw

    return (
        pred_school[:oh, :ow],
        school_prob[:oh, :ow]
    )


# ============================================================
# 7. METRICS
# ============================================================

def binary_counts(
    gt,
    pred
):
    valid = (
        gt != IGNORE_VALUE
    )

    g = gt[valid]
    p = pred[valid]

    tp = int(
        np.sum(
            (g == 1)
            & (p == 1)
        )
    )
    tn = int(
        np.sum(
            (g == 0)
            & (p == 0)
        )
    )
    fp = int(
        np.sum(
            (g == 0)
            & (p == 1)
        )
    )
    fn = int(
        np.sum(
            (g == 1)
            & (p == 0)
        )
    )

    return tp, tn, fp, fn


def metrics_from_counts(
    tp,
    tn,
    fp,
    fn
):
    eps = 1e-12

    precision = tp / (
        tp + fp + eps
    )
    recall = tp / (
        tp + fn + eps
    )
    f1 = (
        2.0 * tp
        / (
            2.0 * tp
            + fp
            + fn
            + eps
        )
    )
    iou = tp / (
        tp
        + fp
        + fn
        + eps
    )
    accuracy = (
        (tp + tn)
        / (
            tp
            + tn
            + fp
            + fn
            + eps
        )
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "accuracy": float(accuracy),
    }


def type_specific_counts(
    type_gt,
    pred,
    target_value
):
    """
    For one target school type:
      target type = positive
      background = negative
      other school types = ignored

    A tile contributes only when it actually contains the target type.
    """
    has_target = bool(
        np.any(
            type_gt == target_value
        )
    )

    if not has_target:
        return None

    valid = (
        (type_gt == 0)
        | (
            type_gt == target_value
        )
    )

    g = (
        type_gt[
            valid
        ]
        == target_value
    ).astype(
        np.uint8
    )

    p = pred[
        valid
    ].astype(
        np.uint8
    )

    tp = int(
        np.sum(
            (g == 1)
            & (p == 1)
        )
    )
    tn = int(
        np.sum(
            (g == 0)
            & (p == 0)
        )
    )
    fp = int(
        np.sum(
            (g == 0)
            & (p == 1)
        )
    )
    fn = int(
        np.sum(
            (g == 1)
            & (p == 0)
        )
    )

    return (
        tp,
        tn,
        fp,
        fn,
    )


def core_boundary_from_valid_mask(
    valid
):
    """
    1-pixel inner boundary of the prepared true-core mask.
    """
    valid = valid.astype(
        bool
    )

    interior = valid.copy()

    interior[1:, :] &= valid[:-1, :]
    interior[:-1, :] &= valid[1:, :]
    interior[:, 1:] &= valid[:, :-1]
    interior[:, :-1] &= valid[:, 1:]

    return (
        valid
        & (~interior)
    )


def evaluate_split(
    model,
    grid_ids,
    device,
    save_dir=None,
    verbose=False
):
    total_tp = 0
    total_tn = 0
    total_fp = 0
    total_fn = 0

    type_totals = {
        key: {
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "positive_tiles": 0,
        }
        for key in TYPE_KEY_TO_LABEL
    }

    rows = []

    if save_dir is not None:

        save_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        pred_dir = save_dir / "school_binary_png"
        prob_dir = save_dir / "school_probability_npy"
        overlay_dir = save_dir / "overlay"
        error_dir = save_dir / "core_error_overlay"

        for d in [
            pred_dir,
            prob_dir,
            overlay_dir,
            error_dir,
        ]:
            d.mkdir(
                parents=True,
                exist_ok=True
            )

    for i, gid in enumerate(
        grid_ids,
        1
    ):
        image_path = find_file(
            IMAGE_DIR,
            gid,
            [".tif", ".tiff"]
        )

        mask_path = find_file(
            MASK_DIR,
            gid,
            [".png"]
        )

        type_mask_path = find_file(
            TYPE_MASK_DIR,
            gid,
            [".png"]
        )

        rgb = read_rgb(
            image_path
        )

        gt = read_mask(
            mask_path,
            gid
        )

        type_gt = read_type_mask(
            type_mask_path
        )

        if (
            rgb.shape[:2] != gt.shape
            or gt.shape != type_gt.shape
        ):
            raise RuntimeError(
                "Image/binary/type-mask shape mismatch for {}"
                .format(
                    gid
                )
            )

        if verbose:
            print(
                "[{}/{}] {}".format(
                    i,
                    len(grid_ids),
                    gid
                )
            )

        pred, school_prob = predict_full_image(
            model,
            rgb,
            device,
            print_windows=verbose
        )

        # Overall binary metrics: same definition as original script.
        tp, tn, fp, fn = binary_counts(
            gt,
            pred
        )

        m = metrics_from_counts(
            tp,
            tn,
            fp,
            fn
        )

        total_tp += tp
        total_tn += tn
        total_fp += fp
        total_fn += fn

        row = {
            "grid_id": gid,
            "valid_pixels": int(
                np.sum(
                    gt != IGNORE_VALUE
                )
            ),
            "gt_school_pixels": int(
                np.sum(
                    gt == 1
                )
            ),
            "gt_primary_pixels": int(
                np.sum(
                    type_gt == 1
                )
            ),
            "gt_secondary_pixels": int(
                np.sum(
                    type_gt == 2
                )
            ),
            "gt_higher_education_pixels": int(
                np.sum(
                    type_gt == 3
                )
            ),
            "pred_school_pixels_on_valid": int(
                np.sum(
                    (pred == 1)
                    & (
                        gt != IGNORE_VALUE
                    )
                )
            ),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "iou": m["iou"],
            "accuracy": m["accuracy"],
        }

        for value, key in TYPE_VALUE_TO_KEY.items():

            counts = type_specific_counts(
                type_gt,
                pred,
                target_value=value,
            )

            if counts is None:

                for suffix in [
                    "TP",
                    "TN",
                    "FP",
                    "FN",
                    "precision",
                    "recall",
                    "f1",
                    "iou",
                ]:
                    row[
                        "{}_{}".format(
                            key,
                            suffix
                        )
                    ] = ""

                continue

            (
                ttp,
                ttn,
                tfp,
                tfn,
            ) = counts

            tm = metrics_from_counts(
                ttp,
                ttn,
                tfp,
                tfn
            )

            type_totals[key]["TP"] += ttp
            type_totals[key]["TN"] += ttn
            type_totals[key]["FP"] += tfp
            type_totals[key]["FN"] += tfn
            type_totals[key]["positive_tiles"] += 1

            row[
                key + "_TP"
            ] = ttp
            row[
                key + "_TN"
            ] = ttn
            row[
                key + "_FP"
            ] = tfp
            row[
                key + "_FN"
            ] = tfn
            row[
                key + "_precision"
            ] = tm["precision"]
            row[
                key + "_recall"
            ] = tm["recall"]
            row[
                key + "_f1"
            ] = tm["f1"]
            row[
                key + "_iou"
            ] = tm["iou"]

        rows.append(
            row
        )

        if save_dir is not None:

            Image.fromarray(
                pred * 255,
                mode="L"
            ).save(
                str(
                    pred_dir
                    / "{}_school.png".format(
                        gid
                    )
                )
            )

            np.save(
                str(
                    prob_dir
                    / "{}_school_probability.npy".format(
                        gid
                    )
                ),
                school_prob.astype(
                    np.float32
                )
            )

            overlay = (
                rgb.astype(
                    np.float32
                )
                .copy()
            )

            idx = (
                pred == 1
            )

            overlay[idx] = (
                (
                    1.0
                    - OVERLAY_ALPHA
                )
                * overlay[idx]
                + OVERLAY_ALPHA
                * SCHOOL_COLOR.astype(
                    np.float32
                )
            )

            overlay = np.clip(
                overlay,
                0,
                255
            ).astype(
                np.uint8
            )

            Image.fromarray(
                overlay,
                mode="RGB"
            ).save(
                str(
                    overlay_dir
                    / "{}_overlay.png".format(
                        gid
                    )
                )
            )

            # Same TP/FP/FN visualization, but the core boundary is now
            # read directly from the prepared 0/1/255 mask.
            core_error = (
                rgb.astype(
                    np.float32
                )
                .copy()
            )

            valid = (
                gt != IGNORE_VALUE
            )

            tp_idx = (
                valid
                & (gt == 1)
                & (pred == 1)
            )
            fp_idx = (
                valid
                & (gt == 0)
                & (pred == 1)
            )
            fn_idx = (
                valid
                & (gt == 1)
                & (pred == 0)
            )

            colors = [
                (
                    tp_idx,
                    np.array(
                        [0, 220, 0],
                        dtype=np.float32
                    )
                ),
                (
                    fp_idx,
                    np.array(
                        [255, 0, 0],
                        dtype=np.float32
                    )
                ),
                (
                    fn_idx,
                    np.array(
                        [0, 120, 255],
                        dtype=np.float32
                    )
                ),
            ]

            for idx_mask, color in colors:

                core_error[
                    idx_mask
                ] = (
                    0.45
                    * core_error[
                        idx_mask
                    ]
                    + 0.55
                    * color
                )

            core_error = np.clip(
                core_error,
                0,
                255
            ).astype(
                np.uint8
            )

            boundary = core_boundary_from_valid_mask(
                valid
            )

            core_error[
                boundary
            ] = np.array(
                [255, 255, 0],
                dtype=np.uint8
            )

            Image.fromarray(
                core_error,
                mode="RGB"
            ).save(
                str(
                    error_dir
                    / "{}_core_error.png".format(
                        gid
                    )
                )
            )

        if verbose:
            print(
                "    IoU={:.4f} P={:.4f} R={:.4f} F1={:.4f}"
                .format(
                    m["iou"],
                    m["precision"],
                    m["recall"],
                    m["f1"],
                )
            )

    micro = metrics_from_counts(
        total_tp,
        total_tn,
        total_fp,
        total_fn
    )

    positive_rows = [
        r
        for r in rows
        if r["gt_school_pixels"] > 0
    ]

    def mean_key(key):
        if not positive_rows:
            return 0.0

        return float(
            np.mean(
                [
                    r[key]
                    for r in positive_rows
                ]
            )
        )

    macro_positive = {
        "precision": mean_key(
            "precision"
        ),
        "recall": mean_key(
            "recall"
        ),
        "f1": mean_key(
            "f1"
        ),
        "iou": mean_key(
            "iou"
        ),
    }

    type_specific = {}

    for key, label_cn in TYPE_KEY_TO_LABEL.items():

        c = type_totals[
            key
        ]

        tm = metrics_from_counts(
            c["TP"],
            c["TN"],
            c["FP"],
            c["FN"],
        )

        type_specific[key] = {
            "label_cn": label_cn,
            "positive_tiles": int(
                c["positive_tiles"]
            ),
            "gt_target_pixels": int(
                c["TP"]
                + c["FN"]
            ),
            "counts": {
                "TP": int(
                    c["TP"]
                ),
                "TN": int(
                    c["TN"]
                ),
                "FP": int(
                    c["FP"]
                ),
                "FN": int(
                    c["FN"]
                ),
            },
            "micro": tm,
            "evaluation_rule": (
                "only tiles containing this target type; "
                "target type=positive; background=negative; "
                "other school types=ignored"
            ),
        }

    return {
        "rows": rows,
        "counts": {
            "TP": total_tp,
            "TN": total_tn,
            "FP": total_fp,
            "FN": total_fn,
        },
        "micro": micro,
        "macro_positive": macro_positive,
        "positive_tiles": len(
            positive_rows
        ),
        "type_specific": type_specific,
    }


# ============================================================
# 8. CSV / CHECKPOINT
# ============================================================

def append_training_log(row):
    fields = [
        "epoch",
        "train_loss",
        "lr",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_iou",
        "val_accuracy",
        "best_val_iou",
    ]

    exists = TRAIN_LOG.exists()

    with open(
        str(TRAIN_LOG),
        "a",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_val_iou
):
    obj = {
        "epoch": int(epoch),
        "best_val_iou": float(
            best_val_iou
        ),
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
        ),
        "settings": {
            "model": PRETRAINED_MODEL,
            "crop_size": CROP_SIZE,
            "stride": INFER_STRIDE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patches_per_image_per_epoch": (
                PATCHES_PER_IMAGE_PER_EPOCH
            ),
            "base_lr": BASE_LR,
            "weight_decay": WEIGHT_DECAY,
            "ignore_value": IGNORE_VALUE,
            "label_semantics": (
                "true core: school=1, all remaining core pixels=0; "
                "outside true core=255 ignore"
            ),
            "core_supervision": (
                "prepared masks_png: core inside=0/1, outside core=255"
            ),
            "dataset_version": DATASET_VERSION,
        },
    }

    torch.save(
        obj,
        str(path)
    )


def load_resume_checkpoint(
    path,
    model,
    optimizer,
    device
):
    obj = torch.load(
        str(path),
        map_location=device
    )

    model.load_state_dict(
        obj[
            "model_state_dict"
        ],
        strict=True
    )

    optimizer.load_state_dict(
        obj[
            "optimizer_state_dict"
        ]
    )

    start_epoch = (
        int(
            obj[
                "epoch"
            ]
        )
        + 1
    )

    best_val_iou = float(
        obj.get(
            "best_val_iou",
            -1.0
        )
    )

    return (
        start_epoch,
        best_val_iou
    )


def write_rows_csv(
    path,
    rows
):
    if not rows:
        return

    fields = list(
        rows[0].keys()
    )

    with open(
        str(path),
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 9. TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch
):
    model.train()

    running_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(
        loader,
        1
    ):
        x, target, gids = batch

        x = x.to(
            device,
            non_blocking=True
        )
        target = target.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        logits = forward_logits(
            model,
            x,
            target.shape[-2:]
        )

        loss = criterion(
            logits,
            target
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite loss at epoch {}, step {}: {}".format(
                    epoch,
                    step,
                    loss.item()
                )
            )

        loss.backward()
        optimizer.step()

        running_loss += float(
            loss.item()
        )
        n_batches += 1

        if step % 20 == 0 or step == len(loader):
            print(
                "\r  epoch {:02d}/{:02d} step {:03d}/{:03d} loss={:.5f}".format(
                    epoch,
                    EPOCHS,
                    step,
                    len(loader),
                    running_loss
                    / max(1, n_batches),
                ),
                end="",
                flush=True
            )

    print()

    return (
        running_loss
        / max(1, n_batches)
    )


# ============================================================
# 10. FOLD TRAINING
# ============================================================

def print_type_specific_summary(
    type_specific,
    prefix=""
):
    for key in [
        "primary",
        "secondary",
        "higher_education",
    ]:

        obj = type_specific[
            key
        ]

        m = obj[
            "micro"
        ]

        print(
            "{}{}: IoU={:.4f} P={:.4f} R={:.4f} F1={:.4f} "
            "(tiles={}, gt_pixels={})"
            .format(
                prefix,
                obj[
                    "label_cn"
                ],
                m[
                    "iou"
                ],
                m[
                    "precision"
                ],
                m[
                    "recall"
                ],
                m[
                    "f1"
                ],
                obj[
                    "positive_tiles"
                ],
                obj[
                    "gt_target_pixels"
                ],
            )
        )


def run_one_fold(
    fold_id,
    device,
):
    configure_fold_paths(
        fold_id
    )

    completed_summary = (
        OUT_ROOT
        / "test_summary.json"
    )

    if (
        SKIP_COMPLETED_FOLDS
        and completed_summary.exists()
    ):
        print()
        print(
            "[Fold {}] already completed; skipping.".format(
                fold_id
            )
        )
        print(
            "  {}".format(
                completed_summary
            )
        )
        return

    # Keep the same seed behavior as the original single-run script.
    set_seed(
        SEED
    )

    paths = get_fold_paths(
        fold_id
    )

    train_ids = read_id_list(
        paths["train"]
    )
    val_ids = read_id_list(
        paths["val"]
    )
    test_ids = read_id_list(
        paths["test"]
    )

    if (
        len(train_ids) == 0
        or len(val_ids) == 0
        or len(test_ids) == 0
    ):
        raise RuntimeError(
            "Fold {} Train/Val/Test must all be non-empty, got {}/{}/{}"
            .format(
                fold_id,
                len(train_ids),
                len(val_ids),
                len(test_ids),
            )
        )

    if (
        set(train_ids) & set(val_ids)
        or set(train_ids) & set(test_ids)
        or set(val_ids) & set(test_ids)
    ):
        raise RuntimeError(
            "Fold {} Train/Val/Test ID overlap detected."
            .format(
                fold_id
            )
        )

    all_ids = (
        list(train_ids)
        + list(val_ids)
        + list(test_ids)
    )

    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError(
            "Fold {} contains duplicate IDs."
            .format(
                fold_id
            )
        )

    print()
    print(
        "=" * 88
    )
    print(
        "FOLD {} | train={} val={} test={}"
        .format(
            fold_id,
            len(train_ids),
            len(val_ids),
            len(test_ids),
        )
    )
    print(
        "=" * 88
    )
    print(
        "Dataset version: {}".format(
            DATASET_VERSION
        )
    )
    print(
        "Output: {}".format(
            OUT_ROOT
        )
    )

    # The data-preparation script has already built the true-core mask.
    # Here we only verify binary/type consistency.
    write_prepared_mask_audit(
        all_ids
    )

    print()
    print("Prepared GT semantics:")
    print(
        "  true core: school=1; all remaining core pixels=0"
    )
    print(
        "  outside true core: 255 ignore/context only"
    )
    print(
        "  type mask: evaluation only; NEVER enters model/loss"
    )

    model = build_model(
        device
    )

    optimizer = make_optimizer(
        model
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=IGNORE_VALUE
    )

    dataset = SchoolCropDataset(
        grid_ids=train_ids,
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        crop_size=CROP_SIZE,
        patches_per_image=PATCHES_PER_IMAGE_PER_EPOCH,
        school_center_prob=SCHOOL_CENTER_PROB,
        min_valid_pixels=MIN_VALID_PIXELS_PER_CROP,
        max_tries=MAX_CROP_TRIES,
        augment=USE_AUGMENTATION,
        cache_size=IMAGE_CACHE_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    print(
        "Training patches/epoch: {}".format(
            len(dataset)
        )
    )
    print(
        "Optimizer: AdamW, initial lr={}, wd={}"
        .format(
            BASE_LR,
            WEIGHT_DECAY
        )
    )

    start_epoch = 1
    best_val_iou = -1.0
    best_epoch = -1

    if (
        RESUME_IF_AVAILABLE
        and LAST_CKPT.exists()
    ):
        (
            start_epoch,
            best_val_iou
        ) = load_resume_checkpoint(
            LAST_CKPT,
            model,
            optimizer,
            device
        )

        print(
            "Resuming Fold {} at epoch {}, best_val_iou={:.6f}"
            .format(
                fold_id,
                start_epoch,
                best_val_iou
            )
        )

    for epoch in range(
        start_epoch,
        EPOCHS + 1
    ):
        t0 = time.time()

        lr = update_poly_lr(
            optimizer,
            epoch - 1
        )

        train_loss = train_one_epoch(
            model,
            loader,
            optimizer,
            criterion,
            device,
            epoch
        )

        val_result = None

        if (
            epoch % VAL_EVERY == 0
            or epoch == EPOCHS
        ):
            print(
                "Validation at epoch {}..."
                .format(
                    epoch
                )
            )

            val_result = evaluate_split(
                model,
                val_ids,
                device,
                save_dir=None,
                verbose=False
            )

            vm = val_result[
                "micro"
            ]

            current_val_iou = float(
                vm[
                    "iou"
                ]
            )

            # SAME model-selection rule as uploaded code:
            # ONLY overall validation micro IoU selects the checkpoint.
            if (
                current_val_iou
                > best_val_iou
            ):
                best_val_iou = (
                    current_val_iou
                )
                best_epoch = epoch

                save_checkpoint(
                    BEST_CKPT,
                    model,
                    optimizer,
                    epoch,
                    best_val_iou
                )

                print(
                    "  NEW BEST: "
                    "P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
                    .format(
                        vm["precision"],
                        vm["recall"],
                        vm["f1"],
                        vm["iou"],
                    )
                )

            else:
                print(
                    "  Val: P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
                    .format(
                        vm["precision"],
                        vm["recall"],
                        vm["f1"],
                        vm["iou"],
                    )
                )

        save_checkpoint(
            LAST_CKPT,
            model,
            optimizer,
            epoch,
            best_val_iou
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": lr,
            "val_precision": "",
            "val_recall": "",
            "val_f1": "",
            "val_iou": "",
            "val_accuracy": "",
            "best_val_iou": best_val_iou,
        }

        if val_result is not None:

            vm = val_result[
                "micro"
            ]

            row.update({
                "val_precision": vm["precision"],
                "val_recall": vm["recall"],
                "val_f1": vm["f1"],
                "val_iou": vm["iou"],
                "val_accuracy": vm["accuracy"],
            })

        append_training_log(
            row
        )

        print(
            "Epoch {:02d} finished in {:.1f}s, "
            "train_loss={:.5f}, lr={:.8f}"
            .format(
                epoch,
                time.time() - t0,
                train_loss,
                lr
            )
        )

    if not BEST_CKPT.exists():
        raise RuntimeError(
            "Fold {} best checkpoint was not created."
            .format(
                fold_id
            )
        )

    best_obj = torch.load(
        str(BEST_CKPT),
        map_location=device
    )

    model.load_state_dict(
        best_obj[
            "model_state_dict"
        ],
        strict=True
    )

    best_epoch = int(
        best_obj[
            "epoch"
        ]
    )
    best_val_iou = float(
        best_obj[
            "best_val_iou"
        ]
    )

    print()
    print(
        "Re-evaluating Fold {} best validation model "
        "(epoch {}) and saving predictions..."
        .format(
            fold_id,
            best_epoch
        )
    )

    val_best = evaluate_split(
        model,
        val_ids,
        device,
        save_dir=VAL_BEST_DIR,
        verbose=False
    )

    best_validation_summary = {
        "dataset_version": DATASET_VERSION,
        "fold_id": int(
            fold_id
        ),
        "model": str(
            PRETRAINED_MODEL
        ),
        "best_epoch": best_epoch,
        "best_val_iou_recorded": best_val_iou,
        "selection_rule": (
            "highest overall validation micro IoU; "
            "type-specific metrics do not affect model selection"
        ),
        "micro": val_best[
            "micro"
        ],
        "macro_positive": val_best[
            "macro_positive"
        ],
        "type_specific": val_best[
            "type_specific"
        ],
        "counts": val_best[
            "counts"
        ],
        "positive_tiles": val_best[
            "positive_tiles"
        ],
        "label_semantics": (
            "prepared masks_png: true core school=1, "
            "remaining core=0, outside core=255"
        ),
        "type_mask_role": (
            "evaluation only; never used as input or supervision"
        ),
    }

    with open(
        str(
            OUT_ROOT
            / "best_validation_summary.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            best_validation_summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        "FINAL TEST Fold {}: best overall validation-IoU checkpoint."
        .format(
            fold_id
        )
    )

    test_result = evaluate_split(
        model,
        test_ids,
        device,
        save_dir=TEST_OUT_DIR,
        verbose=True
    )

    write_rows_csv(
        OUT_ROOT
        / "test_per_image_metrics.csv",
        test_result[
            "rows"
        ]
    )

    test_summary = {
        "dataset_version": DATASET_VERSION,
        "fold_id": int(
            fold_id
        ),
        "model": str(
            PRETRAINED_MODEL
        ),
        "best_epoch": best_epoch,
        "selection_rule": (
            "highest overall validation micro IoU; "
            "held-out test evaluated once"
        ),
        "micro": test_result[
            "micro"
        ],
        "macro_positive": test_result[
            "macro_positive"
        ],
        "type_specific": test_result[
            "type_specific"
        ],
        "counts": test_result[
            "counts"
        ],
        "positive_tiles": test_result[
            "positive_tiles"
        ],
        "type_specific_definition": (
            "same binary prediction; target type=positive, "
            "background=negative, other school types=ignored; "
            "only tiles containing target type contribute"
        ),
        "settings": {
            "crop_size": CROP_SIZE,
            "stride": INFER_STRIDE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patches_per_image_per_epoch": (
                PATCHES_PER_IMAGE_PER_EPOCH
            ),
            "optimizer": "AdamW",
            "base_lr": BASE_LR,
            "weight_decay": WEIGHT_DECAY,
            "poly_power": POLY_POWER,
            "ignore_value": IGNORE_VALUE,
            "label_semantics": (
                "true core school=1, remaining core=0, outside core=255"
            ),
            "school_center_prob": SCHOOL_CENTER_PROB,
            "type_mask_used_for_training": False,
        },
    }

    with open(
        str(
            OUT_ROOT
            / "test_summary.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            test_summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    tm = test_result[
        "micro"
    ]

    with open(
        str(
            OUT_ROOT
            / "test_summary.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "SegFormer-B2 pure-visual binary school-building baseline\n"
        )
        f.write(
            "dataset_version={}\n".format(
                DATASET_VERSION
            )
        )
        f.write(
            "fold_id={}\n".format(
                fold_id
            )
        )
        f.write(
            "best_epoch={}\n".format(
                best_epoch
            )
        )
        f.write(
            "overall_precision={:.8f}\n".format(
                tm["precision"]
            )
        )
        f.write(
            "overall_recall={:.8f}\n".format(
                tm["recall"]
            )
        )
        f.write(
            "overall_f1={:.8f}\n".format(
                tm["f1"]
            )
        )
        f.write(
            "overall_iou={:.8f}\n".format(
                tm["iou"]
            )
        )
        f.write(
            "overall_accuracy={:.8f}\n".format(
                tm["accuracy"]
            )
        )

        for key in [
            "primary",
            "secondary",
            "higher_education",
        ]:

            obj = test_result[
                "type_specific"
            ][key]

            m = obj[
                "micro"
            ]

            for metric in [
                "precision",
                "recall",
                "f1",
                "iou",
                "accuracy",
            ]:
                f.write(
                    "{}_{}={:.8f}\n".format(
                        key,
                        metric,
                        m[metric],
                    )
                )

            f.write(
                "{}_positive_tiles={}\n".format(
                    key,
                    obj[
                        "positive_tiles"
                    ],
                )
            )
            f.write(
                "{}_gt_target_pixels={}\n".format(
                    key,
                    obj[
                        "gt_target_pixels"
                    ],
                )
            )

    print()
    print(
        "=" * 88
    )
    print(
        "FOLD {} FINISHED".format(
            fold_id
        )
    )
    print(
        "=" * 88
    )
    print(
        "Best epoch    : {}".format(
            best_epoch
        )
    )
    print(
        "Val IoU       : {:.4f}".format(
            best_val_iou
        )
    )
    print(
        "Overall Test  : "
        "IoU={:.4f} P={:.4f} R={:.4f} F1={:.4f} Acc={:.4f}"
        .format(
            tm["iou"],
            tm["precision"],
            tm["recall"],
            tm["f1"],
            tm["accuracy"],
        )
    )
    print(
        "Type-specific test metrics:"
    )

    print_type_specific_summary(
        test_result[
            "type_specific"
        ],
        prefix="  "
    )

    print(
        "Output: {}".format(
            OUT_ROOT
        )
    )

    del model
    del optimizer
    del dataset
    del loader

    torch.cuda.empty_cache()


def collect_completed_fold_summaries():
    rows = []

    for fold_id in range(
        5
    ):

        summary_path = (
            OUT_BASE_ROOT
            / "fold_{}".format(
                fold_id
            )
            / "test_summary.json"
        )

        if not summary_path.exists():
            continue

        with open(
            str(summary_path),
            "r",
            encoding="utf-8"
        ) as f:

            obj = json.load(
                f
            )

        overall = obj[
            "micro"
        ]

        row = {
            "fold_id": int(
                fold_id
            ),
            "best_epoch": int(
                obj[
                    "best_epoch"
                ]
            ),
            "overall_precision": overall[
                "precision"
            ],
            "overall_recall": overall[
                "recall"
            ],
            "overall_f1": overall[
                "f1"
            ],
            "overall_iou": overall[
                "iou"
            ],
            "overall_accuracy": overall[
                "accuracy"
            ],
        }

        for key in [
            "primary",
            "secondary",
            "higher_education",
        ]:

            type_obj = obj[
                "type_specific"
            ][key]

            m = type_obj[
                "micro"
            ]

            row.update({
                key + "_precision": m[
                    "precision"
                ],
                key + "_recall": m[
                    "recall"
                ],
                key + "_f1": m[
                    "f1"
                ],
                key + "_iou": m[
                    "iou"
                ],
                key + "_accuracy": m[
                    "accuracy"
                ],
                key + "_positive_tiles": type_obj[
                    "positive_tiles"
                ],
                key + "_gt_target_pixels": type_obj[
                    "gt_target_pixels"
                ],
            })

        rows.append(
            row
        )

    if rows:

        write_rows_csv(
            OUT_BASE_ROOT
            / "completed_folds_test_summary.csv",
            rows
        )


# ============================================================
# 11. MAIN
# ============================================================

def main():
    print(
        "=" * 88
    )
    print(
        "SEGFORMER-B2 PURE-VISUAL BINARY | NEW 5-FOLD SCHOOL DATASET"
    )
    print(
        "=" * 88
    )

    check_inputs()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )

    device = torch.device(
        "cuda:0"
    )

    print(
        "GPU: {}".format(
            torch.cuda.get_device_name(
                0
            )
        )
    )
    print(
        "Dataset version: {}".format(
            DATASET_VERSION
        )
    )
    print(
        "RUN_FOLDS = {}".format(
            RUN_FOLDS
        )
    )
    print(
        "Type mask is evaluation-only; model remains binary."
    )

    for fold_id in RUN_FOLDS:

        run_one_fold(
            int(fold_id),
            device
        )

        collect_completed_fold_summaries()

    print()
    print(
        "=" * 88
    )
    print(
        "REQUESTED FOLDS FINISHED"
    )
    print(
        "=" * 88
    )
    print(
        "Completed-fold table:"
    )
    print(
        "  {}".format(
            OUT_BASE_ROOT
            / "completed_folds_test_summary.csv"
        )
    )


if __name__ == "__main__":
    main()
