# -*- coding: utf-8 -*-
r"""
Fold 0:
current SegFormer + adaptive visual-feature POI + VLM scene recovery
+ conditional AOI outside soft suppression
======================================================================

This script keeps the PREVIOUSLY AGREED AOI role unchanged and only adapts
its input baseline to the current Fold-0 result.

Current baseline:
    SegFormer
      -> empirical-radius POI + visual-feature adaptive lambda(x)
      -> VLM scene-triggered recovery

AOI role:
    model discovers;
    AOI calibrates spatial ownership.

AOI principle
-------------
1) AOI does NOT create school predictions.
2) AOI interior is NEVER suppressed.
3) An AOI becomes active only if the EXISTING current prediction already
   overlaps it by at least MIN_PRED_OVERLAP_PIXELS positive pixels.
4) Only predictions OUTSIDE the union of active AOIs are softly suppressed.
5) If a tile has NO active AOI, the tile is unchanged.

For active AOIs:

    d(x) = 0                                 inside active AOI
           distance to nearest active AOI    outside

    support(x) = exp(-d(x)^2 / (2*sigma^2))

    multiplier(x) = floor + (1-floor)*support(x)

    P_corr(x) = P_base(x)                    inside active AOI
                P_base(x)*multiplier(x)       outside active AOI

Therefore:
    P_corr <= P_base
    AOI can remove positives, but can never create positives.

Protocol
--------
- Fixed threshold = 0.5.
- AOI activation threshold = 16 positive pixels (NOT tuned).
- sigma/floor are selected on Fold-0 VALIDATION only by IoU.
- no-op floor=1.0 is included.
- selected configuration is frozen and evaluated ONCE on TEST.
- current VLM TEST probabilities are reused directly.
- current VLM VALIDATION probabilities are reconstructed once from the
  already trained recovery-gate checkpoint; nothing is retrained.

Required local script beside this file
--------------------------------------
    vlm_recovery_fold0.py

Required current AOI
--------------------
G:\SH\上海市\学校实验区_500m_B150m\0924\学校AOI.shp

Run
---
    conda activate school_vlm
    python aoi_after_vlm_fold0.py

Outputs
-------
E:\larse_result\aoi_after_adaptive_poi_vlm_fold0\

    val_vlm_probability_npy\
    validation_search.csv
    best_config.json
    test_metrics_comparison.csv
    test_per_tile_metrics.csv
    summary.json

    test_outputs\
        corrected_probability_npy\
        corrected_binary_png\
        active_aoi_mask_png\
        outside_distance_npy\
        removed_pixels_png\
"""

from __future__ import print_function

import csv
import json
import math
import importlib.util
from pathlib import Path

import numpy as np
import geopandas as gpd
from PIL import Image

import torch

import tifffile
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
import pyproj
from scipy.ndimage import distance_transform_edt


# =============================================================================
# 1. IMPORT CURRENT VLM MODULE
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def import_local_module(
    module_name,
    filename,
):
    path = SCRIPT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            "Required local script not found:\n{}\n\n"
            "Put this AOI script in the same folder as vlm_after_adaptive_poi_fold0.py."
            .format(path)
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot import local script:\n{}".format(path)
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


REC = import_local_module(
    "current_vlm_fold0_for_aoi",
    "train_vlm_recovery.py",
)


# =============================================================================
# 2. PATHS
# =============================================================================

DATASET_ROOT = REC.DATASET_ROOT
IMAGE_DIR = REC.IMAGE_DIR
MASK_DIR = REC.MASK_DIR

VAL_LIST = REC.VAL_LIST
TEST_LIST = REC.TEST_LIST

SCHOOL_AOI_FILE = SCRIPT_DIR / "data" / "auxiliary" / "学校AOI.shp"

CURRENT_VLM_ROOT = Path(
    r"E:\larse_result\vlm_scene_recovery_adaptive_poi_fold0"
)

CURRENT_VLM_CKPT = (
    CURRENT_VLM_ROOT
    / "checkpoints"
    / "best_recovery_gate.pth"
)

CURRENT_VLM_TEST_PROB = (
    CURRENT_VLM_ROOT
    / "test_predictions"
    / "school_probability_npy"
)

CURRENT_VLM_SUMMARY = (
    CURRENT_VLM_ROOT
    / "summary.json"
)

OUT_ROOT = Path(
    r"E:\larse_result\aoi_after_adaptive_poi_vlm_fold0"
)

VAL_VLM_PROB_DIR = (
    OUT_ROOT
    / "val_vlm_probability_npy"
)

VAL_SEARCH_CSV = (
    OUT_ROOT
    / "validation_search.csv"
)

BEST_CONFIG_JSON = (
    OUT_ROOT
    / "best_config.json"
)

TEST_METRICS_CSV = (
    OUT_ROOT
    / "test_metrics_comparison.csv"
)

TEST_PER_TILE_CSV = (
    OUT_ROOT
    / "test_per_tile_metrics.csv"
)

SUMMARY_JSON = (
    OUT_ROOT
    / "summary.json"
)

TEST_OUT = (
    OUT_ROOT
    / "test_outputs"
)

CORR_PROB_DIR = (
    TEST_OUT
    / "corrected_probability_npy"
)

CORR_BIN_DIR = (
    TEST_OUT
    / "corrected_binary_png"
)

ACTIVE_AOI_DIR = (
    TEST_OUT
    / "active_aoi_mask_png"
)

DIST_DIR = (
    TEST_OUT
    / "outside_distance_npy"
)

REMOVED_DIR = (
    TEST_OUT
    / "removed_pixels_png"
)


# =============================================================================
# 3. SETTINGS -- SAME AOI PRINCIPLE AS PREVIOUS EXP56
# =============================================================================

THRESHOLD = 0.50
IGNORE_VALUE = 255

# Fixed, not tuned.
MIN_PRED_OVERLAP_PIXELS = 16

# Validation-only candidates, unchanged from previous AOI experiment.
SIGMA_CANDIDATES_M = [
    5.0,
    10.0,
    20.0,
    30.0,
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
]

# floor=1.0 is the no-op safeguard.
FLOOR_CANDIDATES = [
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.00,
]

EPS = 1e-12


# =============================================================================
# 4. BASIC HELPERS
# =============================================================================

def ensure_dirs():
    for p in [
        OUT_ROOT,
        VAL_VLM_PROB_DIR,
        TEST_OUT,
        CORR_PROB_DIR,
        CORR_BIN_DIR,
        ACTIVE_AOI_DIR,
        DIST_DIR,
        REMOVED_DIR,
    ]:
        p.mkdir(
            parents=True,
            exist_ok=True,
        )


def check_inputs():
    required = [
        DATASET_ROOT,
        IMAGE_DIR,
        MASK_DIR,
        VAL_LIST,
        TEST_LIST,
        SCHOOL_AOI_FILE,
        CURRENT_VLM_CKPT,
        CURRENT_VLM_TEST_PROB,
        CURRENT_VLM_SUMMARY,
    ]

    for p in required:
        if not Path(p).exists():
            raise FileNotFoundError(
                "Required input not found:\n{}".format(p)
            )


def read_ids(path):
    ids = []

    with open(
        str(path),
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            x = line.strip()

            if x:
                ids.append(x)

    return ids


def find_image(gid):
    return REC.find_file(
        IMAGE_DIR,
        gid,
        REC.IMAGE_SUFFIXES,
    )


def probability_path(
    root,
    gid,
):
    root = Path(root)

    candidates = [
        root
        / (
            str(gid)
            + "_school_probability.npy"
        ),
        root
        / (
            str(gid)
            + ".npy"
        ),
    ]

    for p in candidates:
        if p.exists():
            return p

    # Return canonical path for writing if not found.
    return (
        root
        / (
            str(gid)
            + "_school_probability.npy"
        )
    )


def read_probability(
    root,
    gid,
):
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

    if arr.ndim != 2:
        raise RuntimeError(
            "Probability must be HxW:\n{}".format(p)
        )

    if not np.isfinite(arr).all():
        raise RuntimeError(
            "NaN/Inf in probability:\n{}".format(p)
        )

    return np.clip(
        arr,
        0.0,
        1.0,
    )


def write_csv(
    path,
    rows,
):
    if not rows:
        return

    with open(
        str(path),
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_json(
    path,
    obj,
):
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


# =============================================================================
# 5. METRICS -- CURRENT DATASET: 255 IS IGNORE
# =============================================================================

def counts(
    gt,
    pred,
):
    valid = (
        gt != IGNORE_VALUE
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
        fn,
    )


def add_counts(
    total,
    c,
):
    for i in range(4):
        total[i] += int(
            c[i]
        )


def metrics_from_counts(
    tp,
    tn,
    fp,
    fn,
):
    precision = (
        tp
        / max(
            1,
            tp + fp,
        )
    )

    recall = (
        tp
        / max(
            1,
            tp + fn,
        )
    )

    f1 = (
        2.0
        * tp
        / max(
            EPS,
            2.0 * tp
            + fp
            + fn,
        )
    )

    iou = (
        tp
        / max(
            1,
            tp
            + fp
            + fn,
        )
    )

    accuracy = (
        (
            tp + tn
        )
        / max(
            1,
            tp
            + tn
            + fp
            + fn,
        )
    )

    fpr = (
        fp
        / max(
            1,
            fp + tn,
        )
    )

    return {
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "accuracy": float(accuracy),
        "false_positive_rate": float(fpr),
    }



# =============================================================================
# 6. GEOTIFF HELPERS -- NO RASTERIO
# =============================================================================

def _tag_values(tags, name):
    if name not in tags:
        return None

    v = tags[name].value

    if isinstance(v, tuple):
        return v

    if isinstance(v, list):
        return tuple(v)

    try:
        return tuple(v)
    except TypeError:
        return (v,)


def read_geotiff_info(image_path):
    """
    Read width/height, pixel scale, tie point and EPSG directly from GeoTIFF tags.
    No rasterio is used.
    """
    with tifffile.TiffFile(str(image_path)) as tif:
        page = tif.pages[0]
        tags = page.tags

        scale = _tag_values(
            tags,
            "ModelPixelScaleTag",
        )

        tie = _tag_values(
            tags,
            "ModelTiepointTag",
        )

        keys = _tag_values(
            tags,
            "GeoKeyDirectoryTag",
        )

        width = int(
            page.imagewidth
        )

        height = int(
            page.imagelength
        )

    if scale is None or tie is None:
        raise RuntimeError(
            "GeoTIFF georeferencing tags missing:\n{}".format(
                image_path
            )
        )

    epsg = None

    if keys is not None and len(keys) >= 4:
        nkeys = int(
            keys[3]
        )

        for i in range(nkeys):
            b = 4 + 4 * i

            if b + 3 >= len(keys):
                break

            key_id = int(
                keys[b]
            )

            location = int(
                keys[b + 1]
            )

            count = int(
                keys[b + 2]
            )

            value = int(
                keys[b + 3]
            )

            # ProjectedCSTypeGeoKey
            if (
                key_id == 3072
                and location == 0
                and count == 1
            ):
                epsg = value
                break

            # GeographicTypeGeoKey
            if (
                key_id == 2048
                and location == 0
                and count == 1
            ):
                epsg = value

    if epsg is None:
        raise RuntimeError(
            "Could not determine EPSG from GeoTIFF:\n{}".format(
                image_path
            )
        )

    return {
        "width": width,
        "height": height,
        "scale_x": float(
            scale[0]
        ),
        "scale_y": float(
            scale[1]
        ),
        "tie_i": float(
            tie[0]
        ),
        "tie_j": float(
            tie[1]
        ),
        "tie_x": float(
            tie[3]
        ),
        "tie_y": float(
            tie[4]
        ),
        "epsg": int(
            epsg
        ),
    }


def map_to_pixel(
    x,
    y,
    geo,
):
    col = (
        geo[
            "tie_i"
        ]
        + (
            float(x)
            - geo[
                "tie_x"
            ]
        )
        / geo[
            "scale_x"
        ]
    )

    row = (
        geo[
            "tie_j"
        ]
        + (
            geo[
                "tie_y"
            ]
            - float(y)
        )
        / geo[
            "scale_y"
        ]
    )

    return (
        float(row),
        float(col),
    )


def pixel_to_map(
    row,
    col,
    geo,
):
    x = (
        geo[
            "tie_x"
        ]
        + (
            float(col)
            - geo[
                "tie_i"
            ]
        )
        * geo[
            "scale_x"
        ]
    )

    y = (
        geo[
            "tie_y"
        ]
        - (
            float(row)
            - geo[
                "tie_j"
            ]
        )
        * geo[
            "scale_y"
        ]
    )

    return (
        float(x),
        float(y),
    )


def metre_scales_for_image(
    geo,
):
    image_crs = pyproj.CRS.from_epsg(
        int(
            geo[
                "epsg"
            ]
        )
    )

    width = int(
        geo[
            "width"
        ]
    )

    height = int(
        geo[
            "height"
        ]
    )

    center_col = (
        width - 1
    ) / 2.0

    center_row = (
        height - 1
    ) / 2.0

    x0, y0 = pixel_to_map(
        center_row,
        center_col,
        geo,
    )

    x1, y1 = pixel_to_map(
        center_row,
        center_col + 1.0,
        geo,
    )

    x2, y2 = pixel_to_map(
        center_row + 1.0,
        center_col,
        geo,
    )

    if image_crs.is_geographic:
        geod = pyproj.Geod(
            ellps="WGS84"
        )

        _, _, mpp_x = geod.inv(
            x0,
            y0,
            x1,
            y1,
        )

        _, _, mpp_y = geod.inv(
            x0,
            y0,
            x2,
            y2,
        )

        mpp_x = abs(
            float(
                mpp_x
            )
        )

        mpp_y = abs(
            float(
                mpp_y
            )
        )

    else:
        unit_factor = 1.0

        try:
            if (
                image_crs.axis_info
                and image_crs.axis_info[
                    0
                ].unit_conversion_factor
            ):
                unit_factor = float(
                    image_crs.axis_info[
                        0
                    ].unit_conversion_factor
                )
        except Exception:
            unit_factor = 1.0

        mpp_x = (
            abs(
                float(
                    x1 - x0
                )
            )
            * unit_factor
        )

        mpp_y = (
            abs(
                float(
                    y2 - y0
                )
            )
            * unit_factor
        )

    if (
        not np.isfinite(
            mpp_x
        )
        or not np.isfinite(
            mpp_y
        )
        or mpp_x <= 0.0
        or mpp_y <= 0.0
    ):
        raise RuntimeError(
            "Invalid metres-per-pixel: {}, {}"
            .format(
                mpp_x,
                mpp_y,
            )
        )

    return (
        mpp_x,
        mpp_y,
    )


def image_bounds_polygon(
    geo,
):
    x0, y0 = pixel_to_map(
        0.0,
        0.0,
        geo,
    )

    x1, y1 = pixel_to_map(
        float(
            geo[
                "height"
            ]
        ),
        float(
            geo[
                "width"
            ]
        ),
        geo,
    )

    minx = min(
        x0,
        x1,
    )

    maxx = max(
        x0,
        x1,
    )

    miny = min(
        y0,
        y1,
    )

    maxy = max(
        y0,
        y1,
    )

    return box(
        minx,
        miny,
        maxx,
        maxy,
    )


def _ring_to_pixels(
    ring,
    geo,
):
    pts = []

    for x, y in ring.coords:
        row, col = map_to_pixel(
            x,
            y,
            geo,
        )

        pts.append(
            (
                float(
                    col
                ),
                float(
                    row
                ),
            )
        )

    return pts


def rasterize_geometry_pil(
    geom,
    geo,
):
    """
    Rasterize one Polygon/MultiPolygon/GeometryCollection with PIL.ImageDraw.
    Holes are explicitly erased.
    """
    from PIL import ImageDraw

    h = int(
        geo[
            "height"
        ]
    )

    w = int(
        geo[
            "width"
        ]
    )

    canvas = Image.new(
        "L",
        (
            w,
            h,
        ),
        0,
    )

    draw = ImageDraw.Draw(
        canvas
    )

    def draw_polygon(poly):
        if (
            poly is None
            or poly.is_empty
        ):
            return

        exterior = _ring_to_pixels(
            poly.exterior,
            geo,
        )

        if len(exterior) >= 3:
            draw.polygon(
                exterior,
                fill=1,
                outline=1,
            )

        for interior in poly.interiors:
            hole = _ring_to_pixels(
                interior,
                geo,
            )

            if len(hole) >= 3:
                draw.polygon(
                    hole,
                    fill=0,
                    outline=0,
                )

    if isinstance(
        geom,
        Polygon,
    ):
        draw_polygon(
            geom
        )

    elif isinstance(
        geom,
        MultiPolygon,
    ):
        for poly in geom.geoms:
            draw_polygon(
                poly
            )

    elif isinstance(
        geom,
        GeometryCollection,
    ):
        for part in geom.geoms:
            if isinstance(
                part,
                Polygon,
            ):
                draw_polygon(
                    part
                )

            elif isinstance(
                part,
                MultiPolygon,
            ):
                for poly in part.geoms:
                    draw_polygon(
                        poly
                    )

    else:
        # AOI should be polygonal. Ignore non-polygon geometry safely.
        return np.zeros(
            (
                h,
                w,
            ),
            dtype=bool,
        )

    return (
        np.asarray(
            canvas,
            dtype=np.uint8,
        )
        > 0
    )


# =============================================================================
# 7. AOI CACHE
# =============================================================================

class AOICache:
    def __init__(
        self,
        path,
    ):
        gdf = gpd.read_file(
            str(path)
        )

        if gdf.crs is None:
            raise RuntimeError(
                "School AOI has no CRS:\n{}".format(path)
            )

        gdf = gdf[
            gdf.geometry.notna()
            & (~gdf.geometry.is_empty)
        ].copy()

        try:
            bad = ~gdf.geometry.is_valid

            if bool(bad.any()):
                gdf.loc[
                    bad,
                    "geometry"
                ] = (
                    gdf.loc[
                        bad,
                        "geometry"
                    ]
                    .buffer(0)
                )

                gdf = gdf[
                    gdf.geometry.notna()
                    & (~gdf.geometry.is_empty)
                ].copy()

        except Exception:
            pass

        self.source = gdf
        self.cache = {}

        print(
            "Loaded school AOIs:",
            len(gdf),
        )

        print(
            "School AOI CRS:",
            gdf.crs,
        )

    def in_crs(
        self,
        target_crs,
    ):
        key = str(
            target_crs
        )

        if key not in self.cache:
            self.cache[
                key
            ] = (
                self.source
                .to_crs(
                    target_crs
                )
            )

        return self.cache[
            key
        ]


# =============================================================================
# 7. ACTIVE AOI + OUTSIDE DISTANCE
# =============================================================================

def build_active_aoi_geometry(
    image_path,
    prob,
    aoi_cache,
):
    """
    An AOI becomes active if the CURRENT baseline prediction contains at least
    MIN_PRED_OVERLAP_PIXELS positive pixels inside that AOI.

    AOI activation is based on existing prediction, not GT.

    This implementation uses tifffile + PIL.ImageDraw + pyproj only.
    rasterio is NOT used.
    """
    baseline_pred = (
        prob
        >= THRESHOLD
    )

    h, w = prob.shape

    geo = read_geotiff_info(
        image_path
    )

    if (
        int(
            geo[
                "height"
            ]
        ) != h
        or int(
            geo[
                "width"
            ]
        ) != w
    ):
        raise RuntimeError(
            "Image/probability shape mismatch for {}: "
            "image={}x{}, prob={}x{}"
            .format(
                Path(
                    image_path
                ).stem,
                geo[
                    "height"
                ],
                geo[
                    "width"
                ],
                h,
                w,
            )
        )

    image_crs = pyproj.CRS.from_epsg(
        int(
            geo[
                "epsg"
            ]
        )
    )

    mpp_x, mpp_y = (
        metre_scales_for_image(
            geo
        )
    )

    aoi = aoi_cache.in_crs(
        image_crs
    )

    tile_geom = image_bounds_polygon(
        geo
    )

    try:
        idx = list(
            aoi.sindex.query(
                tile_geom,
                predicate="intersects",
            )
        )

        candidates = aoi.iloc[
            idx
        ].copy()

    except Exception:
        candidates = aoi.loc[
            aoi.geometry.intersects(
                tile_geom
            )
        ].copy()

    if not candidates.empty:
        candidates = candidates.loc[
            candidates.geometry.intersects(
                tile_geom
            )
        ].copy()

    active_mask = np.zeros(
        (
            h,
            w,
        ),
        dtype=bool,
    )

    n_candidate = int(
        len(
            candidates
        )
    )

    n_active = 0
    active_overlap_pixels = []

    for geom in candidates.geometry:
        if (
            geom is None
            or geom.is_empty
        ):
            continue

        clipped = geom.intersection(
            tile_geom
        )

        if (
            clipped is None
            or clipped.is_empty
        ):
            continue

        aoi_mask = rasterize_geometry_pil(
            clipped,
            geo,
        )

        if not np.any(
            aoi_mask
        ):
            continue

        overlap_px = int(
            np.sum(
                aoi_mask
                & baseline_pred
            )
        )

        if (
            overlap_px
            >= MIN_PRED_OVERLAP_PIXELS
        ):
            active_mask |= (
                aoi_mask
            )

            n_active += 1

            active_overlap_pixels.append(
                overlap_px
            )

    if np.any(
        active_mask
    ):
        outside_distance_m = (
            distance_transform_edt(
                ~active_mask,
                sampling=(
                    mpp_y,
                    mpp_x,
                ),
            )
            .astype(
                np.float32
            )
        )

    else:
        outside_distance_m = np.full(
            (
                h,
                w,
            ),
            np.inf,
            dtype=np.float32,
        )

    return {
        "active_aoi_mask": active_mask,
        "outside_distance_m": outside_distance_m,
        "n_candidate_aois": int(
            n_candidate
        ),
        "n_active_aois": int(
            n_active
        ),
        "active_overlap_pixels": (
            active_overlap_pixels
        ),
    }


# =============================================================================
# 8. AOI CORRECTION
# =============================================================================

def correct_probability(
    prob,
    active_aoi_mask,
    outside_distance_m,
    sigma_m,
    floor,
):
    """
    Inside active AOI:
        unchanged

    Outside active AOI:
        P_corr = P_base * [floor + (1-floor)*Gaussian(distance)]

    AOI never creates positives.
    """
    prob = np.asarray(
        prob,
        dtype=np.float32,
    )

    corrected = prob.copy()

    if (
        float(
            floor
        ) >= 1.0
        or not np.any(
            active_aoi_mask
        )
    ):
        return corrected

    outside = (
        ~active_aoi_mask
    )

    d = outside_distance_m[
        outside
    ].astype(
        np.float64
    )

    support = np.exp(
        -(
            d * d
        )
        / (
            2.0
            * float(
                sigma_m
            )
            * float(
                sigma_m
            )
        )
    ).astype(
        np.float32
    )

    multiplier = (
        float(
            floor
        )
        + (
            1.0
            - float(
                floor
            )
        )
        * support
    )

    corrected[
        outside
    ] *= (
        multiplier
    )

    # Explicit no-creation guarantee.
    corrected = np.minimum(
        corrected,
        prob,
    )

    return np.clip(
        corrected,
        0.0,
        1.0,
    ).astype(
        np.float32
    )


# =============================================================================
# 9. RECONSTRUCT CURRENT VLM VALIDATION PROBABILITIES
# =============================================================================

def load_current_vlm_gate(
    device,
):
    ckpt = torch.load(
        str(
            CURRENT_VLM_CKPT
        ),
        map_location="cpu",
        weights_only=False,
    )

    noop = bool(
        ckpt.get(
            "noop",
            False,
        )
    )

    mean = np.asarray(
        ckpt[
            "feature_mean"
        ],
        dtype=np.float32,
    )

    std = np.asarray(
        ckpt[
            "feature_std"
        ],
        dtype=np.float32,
    )

    if noop:
        model = None

    else:
        model = (
            REC.RecoveryGate()
            .to(
                device
            )
        )

        model.load_state_dict(
            ckpt[
                "model_state_dict"
            ],
            strict=True,
        )

        model.eval()

    return (
        ckpt,
        noop,
        mean,
        std,
        model,
    )


def reconstruct_current_vlm_probability(
    gid,
    split_name,
    noop,
    mean,
    std,
    model,
    device,
):
    if split_name == "val":
        seg_root = REC.SEG_VAL_DIR
        scene_root = REC.SCENE_VAL

    elif split_name == "test":
        seg_root = REC.SEG_TEST_DIR
        scene_root = REC.SCENE_TEST

    else:
        raise ValueError(
            split_name
        )

    gt = REC.read_mask(
        gid
    )

    p_seg = REC.read_prob(
        seg_root,
        gid,
    )

    s_poi = REC.empirical_school_prior(
        gid
    )

    if split_name == "val":
        p_base = REC.read_prob(
            REC.ADAPTIVE_VAL_DIR,
            gid,
        )
    elif split_name == "test":
        p_base = REC.read_prob(
            REC.ADAPTIVE_TEST_DIR,
            gid,
        )
    else:
        raise ValueError(
            split_name
        )

    margin, rank = REC.read_scene(
        scene_root,
        gid,
    )

    feat = REC.make_feature_map(
        p_seg,
        s_poi,
        p_base,
        margin,
    )

    cand = REC.candidate_mask(
        gt,
        p_seg,
        s_poi,
        p_base,
        rank,
    )

    p_final = p_base.copy()

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

        g = REC.gate_values(
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
            1.0,
        )

    # VLM is positive-only relative to the adaptive-POI baseline.
    if np.any(
        p_final
        + 1e-7
        < p_base
    ):
        raise RuntimeError(
            "Current VLM monotonicity violated: {}".format(
                gid
            )
        )

    return p_final.astype(
        np.float32
    )


def ensure_validation_probabilities(
    val_ids,
    noop,
    mean,
    std,
    model,
    device,
):
    missing = [
        gid
        for gid in val_ids
        if not probability_path(
            VAL_VLM_PROB_DIR,
            gid,
        ).exists()
    ]

    print()
    print(
        "Current VLM validation probability cache: "
        "{} existing, {} missing."
        .format(
            len(
                val_ids
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
        1,
    ):
        prob = reconstruct_current_vlm_probability(
            gid=gid,
            split_name="val",
            noop=noop,
            mean=mean,
            std=std,
            model=model,
            device=device,
        )

        np.save(
            str(
                probability_path(
                    VAL_VLM_PROB_DIR,
                    gid,
                )
            ),
            prob,
        )

        if (
            i % 10 == 0
            or i == len(
                missing
            )
        ):
            print(
                "  VAL-VLM [{}/{}] {}"
                .format(
                    i,
                    len(
                        missing
                    ),
                    gid,
                )
            )


# =============================================================================
# 10. BASELINE CONSISTENCY CHECKS
# =============================================================================

def metrics_for_probability_split(
    ids,
    prob_root,
):
    total = [
        0,
        0,
        0,
        0,
    ]

    for gid in ids:
        gt = REC.read_mask(
            gid
        )

        prob = read_probability(
            prob_root,
            gid,
        )

        pred = (
            prob
            >= THRESHOLD
        ).astype(
            np.uint8
        )

        add_counts(
            total,
            counts(
                gt,
                pred,
            ),
        )

    return metrics_from_counts(
        *total
    )


def verify_current_vlm_baselines(
    val_ids,
    test_ids,
):
    summary = json.loads(
        CURRENT_VLM_SUMMARY.read_text(
            encoding="utf-8"
        )
    )

    val_m = metrics_for_probability_split(
        val_ids,
        VAL_VLM_PROB_DIR,
    )

    expected_val_iou = float(
        summary[
            "best_val_iou"
        ]
    )

    if abs(
        float(
            val_m[
                "iou"
            ]
        )
        - expected_val_iou
    ) > 1e-6:
        raise RuntimeError(
            "Reconstructed VLM validation IoU does not match "
            "the completed VLM run.\n"
            "current={:.10f}, expected={:.10f}"
            .format(
                val_m[
                    "iou"
                ],
                expected_val_iou,
            )
        )

    test_m = metrics_for_probability_split(
        test_ids,
        CURRENT_VLM_TEST_PROB,
    )

    expected_test = (
        summary[
            "test_metrics"
        ][
            "Adaptive visual-feature POI + VLM scene recovery"
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
            test_m[
                metric
            ]
        )

        b = float(
            expected_test[
                metric
            ]
        )

        if abs(
            a - b
        ) > 1e-6:
            raise RuntimeError(
                "Current VLM TEST baseline mismatch: "
                "{} current={} expected={}"
                .format(
                    metric,
                    a,
                    b,
                )
            )

    print()
    print(
        "Current VLM baseline checks: PASS"
    )

    print(
        "  VAL  P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(
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
        )
    )

    print(
        "  TEST P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(
            test_m[
                "precision"
            ],
            test_m[
                "recall"
            ],
            test_m[
                "f1"
            ],
            test_m[
                "iou"
            ],
        )
    )

    return (
        val_m,
        test_m,
    )


# =============================================================================
# 11. BUILD COMPACT AOI SEARCH CACHE
# =============================================================================

def build_search_cache(
    ids,
    pred_root,
    aoi_cache,
    split_name,
):
    """
    AOI can only remove CURRENT positive predictions.
    Therefore the validation search stores only current positive pixels.
    """
    items = []

    baseline_total = [
        0,
        0,
        0,
        0,
    ]

    total_candidate_aois = 0
    total_active_aois = 0
    tiles_with_active = 0

    print()
    print(
        "Building {} AOI cache..."
        .format(
            split_name
        )
    )

    for i, gid in enumerate(
        ids,
        1,
    ):
        image_path = find_image(
            gid
        )

        gt = REC.read_mask(
            gid
        )

        prob = read_probability(
            pred_root,
            gid,
        )

        if gt.shape != prob.shape:
            raise RuntimeError(
                "GT/prob shape mismatch: {} {} vs {}"
                .format(
                    gid,
                    gt.shape,
                    prob.shape,
                )
            )

        pred = (
            prob
            >= THRESHOLD
        )

        add_counts(
            baseline_total,
            counts(
                gt,
                pred.astype(
                    np.uint8
                ),
            ),
        )

        geo = build_active_aoi_geometry(
            image_path=image_path,
            prob=prob,
            aoi_cache=aoi_cache,
        )

        valid = (
            gt
            != IGNORE_VALUE
        )

        pos = (
            valid
            & pred
        )

        items.append({
            "gid": gid,
            "pos_prob": (
                prob[
                    pos
                ]
                .astype(
                    np.float32
                )
            ),
            "pos_gt": (
                gt[
                    pos
                ]
                == 1
            ),
            "pos_inside_active": (
                geo[
                    "active_aoi_mask"
                ][
                    pos
                ]
                .astype(
                    bool
                )
            ),
            "pos_distance_m": (
                geo[
                    "outside_distance_m"
                ][
                    pos
                ]
                .astype(
                    np.float32
                )
            ),
            "has_active": bool(
                geo[
                    "n_active_aois"
                ] > 0
            ),
            "n_candidate_aois": int(
                geo[
                    "n_candidate_aois"
                ]
            ),
            "n_active_aois": int(
                geo[
                    "n_active_aois"
                ]
            ),
        })

        total_candidate_aois += int(
            geo[
                "n_candidate_aois"
            ]
        )

        total_active_aois += int(
            geo[
                "n_active_aois"
            ]
        )

        if (
            geo[
                "n_active_aois"
            ] > 0
        ):
            tiles_with_active += 1

        if (
            i % 10 == 0
            or i == len(
                ids
            )
        ):
            print(
                "  {} [{}/{}] {} | active AOI={}"
                .format(
                    split_name,
                    i,
                    len(
                        ids
                    ),
                    gid,
                    geo[
                        "n_active_aois"
                    ],
                )
            )

    return {
        "items": items,
        "baseline_counts": baseline_total,
        "candidate_aois": int(
            total_candidate_aois
        ),
        "active_aois": int(
            total_active_aois
        ),
        "tiles_with_active": int(
            tiles_with_active
        ),
    }


# =============================================================================
# 12. VALIDATION CONFIG EVALUATION
# =============================================================================

def removed_mask_for_item(
    item,
    sigma_m,
    floor,
):
    n = item[
        "pos_prob"
    ].size

    if (
        n == 0
        or not item[
            "has_active"
        ]
        or float(
            floor
        ) >= 1.0
    ):
        return np.zeros(
            (
                n,
            ),
            dtype=bool,
        )

    outside = (
        ~item[
            "pos_inside_active"
        ]
    )

    if not np.any(
        outside
    ):
        return np.zeros(
            (
                n,
            ),
            dtype=bool,
        )

    d = item[
        "pos_distance_m"
    ].astype(
        np.float64
    )

    support = np.zeros(
        d.shape,
        dtype=np.float64,
    )

    finite = np.isfinite(
        d
    )

    support[
        finite
    ] = np.exp(
        -(
            d[
                finite
            ]
            ** 2
        )
        / (
            2.0
            * float(
                sigma_m
            )
            * float(
                sigma_m
            )
        )
    )

    multiplier = (
        float(
            floor
        )
        + (
            1.0
            - float(
                floor
            )
        )
        * support
    )

    corrected = (
        item[
            "pos_prob"
        ].astype(
            np.float64
        )
        * multiplier
    )

    removed = (
        outside
        & (
            corrected
            < THRESHOLD
        )
    )

    return removed


def evaluate_config(
    split_cache,
    sigma_m,
    floor,
):
    (
        base_tp,
        base_tn,
        base_fp,
        base_fn,
    ) = [
        int(x)
        for x in split_cache[
            "baseline_counts"
        ]
    ]

    removed_tp = 0
    removed_fp = 0

    for item in split_cache[
        "items"
    ]:
        removed = removed_mask_for_item(
            item,
            sigma_m,
            floor,
        )

        if not np.any(
            removed
        ):
            continue

        gt_pos = item[
            "pos_gt"
        ]

        removed_tp += int(
            np.sum(
                removed
                & gt_pos
            )
        )

        removed_fp += int(
            np.sum(
                removed
                & (
                    ~gt_pos
                )
            )
        )

    tp = (
        base_tp
        - removed_tp
    )

    fn = (
        base_fn
        + removed_tp
    )

    fp = (
        base_fp
        - removed_fp
    )

    tn = (
        base_tn
        + removed_fp
    )

    m = metrics_from_counts(
        tp,
        tn,
        fp,
        fn,
    )

    m.update({
        "removed_fp_pixels": int(
            removed_fp
        ),
        "removed_tp_pixels": int(
            removed_tp
        ),
        "removed_total_prediction_pixels": int(
            removed_fp
            + removed_tp
        ),
    })

    return m


def is_better(
    a,
    b,
):
    if b is None:
        return True

    # Same Exp56 selection rule.
    for key in [
        "iou",
        "f1",
        "precision",
        "recall",
    ]:
        av = float(
            a[
                key
            ]
        )

        bv = float(
            b[
                key
            ]
        )

        if (
            av
            > bv
            + 1e-12
        ):
            return True

        if (
            av
            < bv
            - 1e-12
        ):
            return False

    # Exact metric tie -> gentler suppression.
    if (
        float(
            a[
                "floor"
            ]
        )
        > float(
            b[
                "floor"
            ]
        )
        + 1e-12
    ):
        return True

    if (
        float(
            a[
                "floor"
            ]
        )
        < float(
            b[
                "floor"
            ]
        )
        - 1e-12
    ):
        return False

    return (
        float(
            a[
                "sigma_m"
            ]
        )
        > float(
            b[
                "sigma_m"
            ]
        )
    )


def search_validation(
    val_cache,
):
    rows = []
    best = None

    total = (
        len(
            SIGMA_CANDIDATES_M
        )
        * len(
            FLOOR_CANDIDATES
        )
    )

    idx = 0

    base = metrics_from_counts(
        *val_cache[
            "baseline_counts"
        ]
    )

    print()
    print(
        "=" * 106
    )
    print(
        "VALIDATION SEARCH: CURRENT VLM + CONDITIONAL AOI OUTSIDE SUPPRESSION"
    )
    print(
        "=" * 106
    )

    print(
        "Current VLM validation baseline: "
        "P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(
            base[
                "precision"
            ],
            base[
                "recall"
            ],
            base[
                "f1"
            ],
            base[
                "iou"
            ],
        )
    )

    for sigma_m in SIGMA_CANDIDATES_M:
        for floor in FLOOR_CANDIDATES:
            idx += 1

            m = evaluate_config(
                val_cache,
                sigma_m,
                floor,
            )

            row = {
                "sigma_m": float(
                    sigma_m
                ),
                "floor": float(
                    floor
                ),
                **m,
            }

            rows.append(
                row
            )

            if is_better(
                row,
                best,
            ):
                best = dict(
                    row
                )

            print(
                "  [{:03d}/{:03d}] sigma={:>6.1f} floor={:.2f} | "
                "P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f} | "
                "removed FP={} TP={}"
                .format(
                    idx,
                    total,
                    sigma_m,
                    floor,
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
                        "removed_fp_pixels"
                    ],
                    m[
                        "removed_tp_pixels"
                    ],
                )
            )

    write_csv(
        VAL_SEARCH_CSV,
        rows,
    )

    if best is None:
        raise RuntimeError(
            "Validation search produced no result."
        )

    config = {
        "selection_split": "validation",
        "threshold_fixed": float(
            THRESHOLD
        ),
        "min_pred_overlap_pixels_for_aoi_activation": int(
            MIN_PRED_OVERLAP_PIXELS
        ),
        "mechanism": (
            "active AOI interior unchanged; "
            "only outside active AOI Gaussian distance suppression; "
            "no active AOI -> unchanged"
        ),
        "sigma_m": float(
            best[
                "sigma_m"
            ]
        ),
        "floor": float(
            best[
                "floor"
            ]
        ),
        "validation_metrics": {
            k: best[
                k
            ]
            for k in [
                "precision",
                "recall",
                "f1",
                "iou",
                "accuracy",
                "false_positive_rate",
                "removed_fp_pixels",
                "removed_tp_pixels",
            ]
        },
    }

    write_json(
        BEST_CONFIG_JSON,
        config,
    )

    print()
    print(
        "BEST VALIDATION CONFIG:"
    )

    print(
        "  sigma_m =",
        best[
            "sigma_m"
        ],
    )

    print(
        "  floor   =",
        best[
            "floor"
        ],
    )

    print(
        "  P={:.4f} R={:.4f} F1={:.4f} IoU={:.4f}"
        .format(
            best[
                "precision"
            ],
            best[
                "recall"
            ],
            best[
                "f1"
            ],
            best[
                "iou"
            ],
        )
    )

    return best


# =============================================================================
# 13. TEST ONCE + SAVE
# =============================================================================

def evaluate_test_once(
    test_ids,
    aoi_cache,
    sigma_m,
    floor,
):
    base_total = [
        0,
        0,
        0,
        0,
    ]

    corr_total = [
        0,
        0,
        0,
        0,
    ]

    removed_fp_total = 0
    removed_tp_total = 0

    active_aois_total = 0
    candidate_aois_total = 0
    tiles_with_active = 0

    per_tile_rows = []

    print()
    print(
        "=" * 106
    )
    print(
        "HELD-OUT TEST: ONCE"
    )
    print(
        "=" * 106
    )

    for i, gid in enumerate(
        test_ids,
        1,
    ):
        image_path = find_image(
            gid
        )

        gt = REC.read_mask(
            gid
        )

        prob = read_probability(
            CURRENT_VLM_TEST_PROB,
            gid,
        )

        geo = build_active_aoi_geometry(
            image_path=image_path,
            prob=prob,
            aoi_cache=aoi_cache,
        )

        corrected = correct_probability(
            prob=prob,
            active_aoi_mask=geo[
                "active_aoi_mask"
            ],
            outside_distance_m=geo[
                "outside_distance_m"
            ],
            sigma_m=sigma_m,
            floor=floor,
        )

        base_pred = (
            prob
            >= THRESHOLD
        ).astype(
            np.uint8
        )

        corr_pred = (
            corrected
            >= THRESHOLD
        ).astype(
            np.uint8
        )

        # AOI may remove positives but may not create them.
        if np.any(
            (base_pred == 0)
            & (
                corr_pred == 1
            )
        ):
            raise RuntimeError(
                "AOI created a new positive: {}".format(
                    gid
                )
            )

        base_c = counts(
            gt,
            base_pred,
        )

        corr_c = counts(
            gt,
            corr_pred,
        )

        add_counts(
            base_total,
            base_c,
        )

        add_counts(
            corr_total,
            corr_c,
        )

        valid = (
            gt
            != IGNORE_VALUE
        )

        removed = (
            valid
            & (
                base_pred == 1
            )
            & (
                corr_pred == 0
            )
        )

        removed_fp = int(
            np.sum(
                removed
                & (
                    gt == 0
                )
            )
        )

        removed_tp = int(
            np.sum(
                removed
                & (
                    gt == 1
                )
            )
        )

        removed_fp_total += (
            removed_fp
        )

        removed_tp_total += (
            removed_tp
        )

        candidate_aois_total += int(
            geo[
                "n_candidate_aois"
            ]
        )

        active_aois_total += int(
            geo[
                "n_active_aois"
            ]
        )

        if (
            geo[
                "n_active_aois"
            ] > 0
        ):
            tiles_with_active += 1

        bm = metrics_from_counts(
            *base_c
        )

        cm = metrics_from_counts(
            *corr_c
        )

        per_tile_rows.append({
            "grid_id": gid,
            "n_candidate_aois": int(
                geo[
                    "n_candidate_aois"
                ]
            ),
            "n_active_aois": int(
                geo[
                    "n_active_aois"
                ]
            ),
            "base_precision": bm[
                "precision"
            ],
            "base_recall": bm[
                "recall"
            ],
            "base_f1": bm[
                "f1"
            ],
            "base_iou": bm[
                "iou"
            ],
            "aoi_precision": cm[
                "precision"
            ],
            "aoi_recall": cm[
                "recall"
            ],
            "aoi_f1": cm[
                "f1"
            ],
            "aoi_iou": cm[
                "iou"
            ],
            "removed_fp_pixels": int(
                removed_fp
            ),
            "removed_tp_pixels": int(
                removed_tp
            ),
        })

        np.save(
            str(
                CORR_PROB_DIR
                / (
                    str(gid)
                    + "_school_probability.npy"
                )
            ),
            corrected.astype(
                np.float32
            ),
        )

        Image.fromarray(
            (
                corr_pred
                * 255
            ).astype(
                np.uint8
            ),
            mode="L",
        ).save(
            str(
                CORR_BIN_DIR
                / (
                    str(gid)
                    + "_school.png"
                )
            )
        )

        Image.fromarray(
            (
                geo[
                    "active_aoi_mask"
                ]
                .astype(
                    np.uint8
                )
                * 255
            ),
            mode="L",
        ).save(
            str(
                ACTIVE_AOI_DIR
                / (
                    str(gid)
                    + "_active_aoi.png"
                )
            )
        )

        np.save(
            str(
                DIST_DIR
                / (
                    str(gid)
                    + "_outside_distance_m.npy"
                )
            ),
            geo[
                "outside_distance_m"
            ].astype(
                np.float32
            ),
        )

        removed_vis = np.zeros(
            gt.shape,
            dtype=np.uint8,
        )

        removed_vis[
            removed
            & (
                gt == 0
            )
        ] = 1

        removed_vis[
            removed
            & (
                gt == 1
            )
        ] = 2

        Image.fromarray(
            removed_vis,
            mode="L",
        ).save(
            str(
                REMOVED_DIR
                / (
                    str(gid)
                    + "_removed_pixels.png"
                )
            )
        )

        if (
            i % 10 == 0
            or i == len(
                test_ids
            )
        ):
            print(
                "  TEST [{}/{}] {} | active AOI={} | "
                "removed FP={} TP={}"
                .format(
                    i,
                    len(
                        test_ids
                    ),
                    gid,
                    geo[
                        "n_active_aois"
                    ],
                    removed_fp,
                    removed_tp,
                )
            )

    base_m = metrics_from_counts(
        *base_total
    )

    corr_m = metrics_from_counts(
        *corr_total
    )

    rows = [
        {
            "method": (
                "Adaptive visual-feature POI + VLM scene recovery"
            ),
            **base_m,
        },
        {
            "method": (
                "Adaptive visual-feature POI + VLM + AOI outside suppression"
            ),
            **corr_m,
        },
        {
            "method": "Delta",
            "TP": (
                corr_m[
                    "TP"
                ]
                - base_m[
                    "TP"
                ]
            ),
            "TN": (
                corr_m[
                    "TN"
                ]
                - base_m[
                    "TN"
                ]
            ),
            "FP": (
                corr_m[
                    "FP"
                ]
                - base_m[
                    "FP"
                ]
            ),
            "FN": (
                corr_m[
                    "FN"
                ]
                - base_m[
                    "FN"
                ]
            ),
            "precision": (
                corr_m[
                    "precision"
                ]
                - base_m[
                    "precision"
                ]
            ),
            "recall": (
                corr_m[
                    "recall"
                ]
                - base_m[
                    "recall"
                ]
            ),
            "f1": (
                corr_m[
                    "f1"
                ]
                - base_m[
                    "f1"
                ]
            ),
            "iou": (
                corr_m[
                    "iou"
                ]
                - base_m[
                    "iou"
                ]
            ),
            "accuracy": (
                corr_m[
                    "accuracy"
                ]
                - base_m[
                    "accuracy"
                ]
            ),
            "false_positive_rate": (
                corr_m[
                    "false_positive_rate"
                ]
                - base_m[
                    "false_positive_rate"
                ]
            ),
        },
    ]

    write_csv(
        TEST_METRICS_CSV,
        rows,
    )

    write_csv(
        TEST_PER_TILE_CSV,
        per_tile_rows,
    )

    summary = {
        "fold_id": 0,
        "diagnostic_fold0": True,
        "base_model": (
            "Adaptive visual-feature POI + VLM scene recovery"
        ),
        "aoi_file": str(
            SCHOOL_AOI_FILE
        ),
        "threshold_fixed": float(
            THRESHOLD
        ),
        "aoi_activation": {
            "condition": (
                "existing baseline threshold-0.5 prediction overlaps AOI"
            ),
            "min_overlap_pixels": int(
                MIN_PRED_OVERLAP_PIXELS
            ),
        },
        "selected_sigma_m": float(
            sigma_m
        ),
        "selected_floor": float(
            floor
        ),
        "mechanism": (
            "active AOI interior unchanged; only outside active AOI "
            "softly suppressed; AOI cannot add predictions; "
            "no active AOI -> unchanged"
        ),
        "baseline_test_metrics": base_m,
        "corrected_test_metrics": corr_m,
        "delta": {
            key: float(
                corr_m[
                    key
                ]
                - base_m[
                    key
                ]
            )
            for key in [
                "precision",
                "recall",
                "f1",
                "iou",
                "accuracy",
                "false_positive_rate",
            ]
        },
        "removed_fp_pixels": int(
            removed_fp_total
        ),
        "removed_tp_pixels": int(
            removed_tp_total
        ),
        "candidate_aois": int(
            candidate_aois_total
        ),
        "active_aois": int(
            active_aois_total
        ),
        "tiles_with_active_aoi": int(
            tiles_with_active
        ),
        "n_test_tiles": int(
            len(
                test_ids
            )
        ),
    }

    write_json(
        SUMMARY_JSON,
        summary,
    )

    print()
    print(
        "=" * 106
    )
    print(
        "FINAL TEST COMPARISON"
    )
    print(
        "=" * 106
    )

    print(
        "{:<50s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}"
        .format(
            "METHOD",
            "P",
            "R",
            "F1",
            "IoU",
            "ACC",
            "FPR",
        )
    )

    for name, m in [
        (
            "Adaptive POI + VLM",
            base_m,
        ),
        (
            "Adaptive POI + VLM + AOI",
            corr_m,
        ),
    ]:
        print(
            "{:<50s} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f} {:>8.4f}"
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
                m[
                    "false_positive_rate"
                ],
            )
        )

    print()
    print(
        "Removed FP pixels (beneficial):",
        removed_fp_total,
    )

    print(
        "Removed TP pixels (harmful):",
        removed_tp_total,
    )

    print(
        "Active AOIs:",
        active_aois_total,
    )

    print(
        "Tiles with >=1 active AOI: {}/{}"
        .format(
            tiles_with_active,
            len(
                test_ids
            ),
        )
    )

    return (
        base_m,
        corr_m,
        summary,
    )


# =============================================================================
# 14. MAIN
# =============================================================================

def main():
    ensure_dirs()
    check_inputs()

    # Important: importing the current VLM script does not run its main(),
    # therefore initialize its empirical-radius POI configuration explicitly.
    REC.load_poi_stage_config()

    val_ids = read_ids(
        VAL_LIST
    )

    test_ids = read_ids(
        TEST_LIST
    )

    if (
        len(
            val_ids
        ) != 42
        or len(
            test_ids
        ) != 84
    ):
        raise RuntimeError(
            "Expected current Fold-0 VAL/TEST = 42/84, got {}/{}."
            .format(
                len(
                    val_ids
                ),
                len(
                    test_ids
                ),
            )
        )

    if set(
        val_ids
    ) & set(
        test_ids
    ):
        raise RuntimeError(
            "Validation/test overlap detected."
        )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 106
    )

    print(
        "FOLD 0: ADAPTIVE VISUAL-FEATURE POI + VLM -> AOI OUTSIDE SOFT SUPPRESSION"
    )

    print(
        "=" * 106
    )

    print(
        "VAL / TEST = {} / {}"
        .format(
            len(
                val_ids
            ),
            len(
                test_ids
            ),
        )
    )

    print(
        "Threshold:",
        THRESHOLD,
    )

    print(
        "AOI activation: >= {} current-positive pixels inside AOI"
        .format(
            MIN_PRED_OVERLAP_PIXELS
        )
    )

    print(
        "AOI rule: active-AOI interior unchanged; "
        "ONLY outside existing positives are suppressed."
    )

    print(
        "No active AOI -> tile unchanged."
    )

    print(
        "Device used only to reconstruct VLM validation probabilities:",
        device,
    )

    (
        ckpt,
        noop,
        mean,
        std,
        recovery_gate,
    ) = load_current_vlm_gate(
        device
    )

    print(
        "Current VLM selected epoch:",
        int(
            ckpt.get(
                "epoch",
                0,
            )
        ),
    )

    # -------------------------------------------------------------------------
    # Reconstruct validation probabilities only once.
    # -------------------------------------------------------------------------
    ensure_validation_probabilities(
        val_ids=val_ids,
        noop=noop,
        mean=mean,
        std=std,
        model=recovery_gate,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Verify current TEST probability files exist.
    # -------------------------------------------------------------------------
    missing_test = [
        gid
        for gid in test_ids
        if not probability_path(
            CURRENT_VLM_TEST_PROB,
            gid,
        ).exists()
    ]

    if missing_test:
        raise RuntimeError(
            "Current VLM TEST probabilities incomplete: {} missing.\n"
            "Run vlm_recovery_fold0.py to completion first."
            .format(
                len(
                    missing_test
                )
            )
        )

    # Hard check that AOI starts from the exact completed VLM baseline.
    verify_current_vlm_baselines(
        val_ids,
        test_ids,
    )

    if recovery_gate is not None:
        del recovery_gate

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Load current school AOI.
    # -------------------------------------------------------------------------
    print()
    print(
        "Loading school AOIs:"
    )

    print(
        " ",
        SCHOOL_AOI_FILE,
    )

    aoi_cache = AOICache(
        SCHOOL_AOI_FILE
    )

    # -------------------------------------------------------------------------
    # Validation-only AOI parameter search.
    # -------------------------------------------------------------------------
    val_cache = build_search_cache(
        ids=val_ids,
        pred_root=VAL_VLM_PROB_DIR,
        aoi_cache=aoi_cache,
        split_name="VAL",
    )

    best = search_validation(
        val_cache
    )

    # -------------------------------------------------------------------------
    # Held-out TEST once.
    # -------------------------------------------------------------------------
    evaluate_test_once(
        test_ids=test_ids,
        aoi_cache=aoi_cache,
        sigma_m=float(
            best[
                "sigma_m"
            ]
        ),
        floor=float(
            best[
                "floor"
            ]
        ),
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
