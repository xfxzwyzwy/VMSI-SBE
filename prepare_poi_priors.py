# -*- coding: utf-8 -*-
r"""
一次性预计算：全部原始学校 POI 的类型化空间先验缓存
================================================

目的
----
把之前 POI 方法中最耗时的“POI -> 每张影像的 Gaussian spatial prior”
单独做完，并保存到磁盘。

本脚本：
1) 直接读取原始 学校POI.shp，不做有效/无效筛选；
2) 仅按 小学 / 中学 / 高等教育 三类分组；
3) 对当前 421 张 prepared images 全部预计算一次；
4) 每张影像、每种 POI 类型只构建 ONE 个 base Gaussian prior；
5) 保存为压缩 NPZ，后续 Fold 0~4 直接读取，不再重复构建。

为什么只需一个 base prior
------------------------
旧的 POI prior 为：

    exp(-d^2 / (2*sigma^2))

且多个 POI 用 pixel-wise max 合并。

若先构建 base_sigma 下的 prior：

    P_base = exp(-d^2 / (2*base_sigma^2))

则任意更小 sigma 可以精确转换为：

    P_sigma = P_base ** ((base_sigma / sigma)^2)

因此无需为每个候选 sigma 重新逐 POI 绘制 Gaussian。

本脚本沿用此前 type-aware POI 实验的最大 school sigma = 2000 m，
因此：
    SCHOOL_BASE_SIGMA_M = 2000.0

后续可支持的候选 sigma：
    75, 100, ..., 2000 m

重要
----
- 这是数据级 cache，与 Fold 无关；
- 一次生成后，Fold 0~4 共用；
- 不读取 train/val/test；
- 不读取 SegFormer 概率；
- 不做任何参数搜索；
- 不做任何精度评价；
- 不修改原 POI / AOI / building 文件；
- POI 直接来源于原始 学校POI.shp，不使用任何有效性筛选结果；
- 支持中断续跑：已存在且通过检查的 tile cache 会跳过。

输入
----
G:\SH\上海市\学校实验区_500m_B150m\0924\
    学校POI.shp
    segformer_binary_5fold_dataset\
        images\
        masks_png\
        metadata.csv

输出
----
E:\larse_result\poi_all_school_base_prior_cache\

    cache\
        <grid_id>.npz
        ...

每个 NPZ：
    school_base:
        shape = [3, H, W], float32
        channel:
            0 primary
            1 middle
            2 higher

    local_poi_counts:
        shape = [3], int32

    mpp_xy:
        [metres_per_pixel_x, metres_per_pixel_y]

    base_sigma_m:
        2000.0

    type_names:
        primary / middle / higher

另外输出：
    cache_manifest.csv
    poi_type_audit.csv
    cache_summary.json

依赖
----
numpy
pandas
geopandas
pyproj
Pillow
tifffile
"""

from __future__ import print_function

import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from PIL import Image
import tifffile
import pyproj

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)

Image.MAX_IMAGE_PIXELS = None


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent / "data" / "auxiliary"
DATASET_ROOT = Path(__file__).resolve().parent / "data"

IMAGE_DIR = (
    DATASET_ROOT
    / "images"
)

MASK_DIR = (
    DATASET_ROOT
    / "masks_png"
)

METADATA_CSV = (
    DATASET_ROOT
    / "metadata.csv"
)

ALL_SCHOOL_POI_SHP = (
    BASE_DIR
    / "学校POI.shp"
)

OUT_ROOT = Path(__file__).resolve().parent / "work" / "poi_priors"
CACHE_DIR = (
    OUT_ROOT
    / "cache"
)

MANIFEST_CSV = (
    OUT_ROOT
    / "cache_manifest.csv"
)

POI_TYPE_AUDIT_CSV = (
    OUT_ROOT
    / "poi_type_audit.csv"
)

SUMMARY_JSON = (
    OUT_ROOT
    / "cache_summary.json"
)


# ============================================================
# 2. SETTINGS
# ============================================================

SCHOOL_TYPE_FIELD = "中类"

TYPE_NAMES = [
    "primary",
    "middle",
    "higher",
]

TYPE_LABELS_CN = {
    "primary": "小学",
    "middle": "中学",
    "higher": "高等教育",
}

PRIMARY_KEYWORDS = [
    "小学",
    "primary",
    "elementary",
]

MIDDLE_KEYWORDS = [
    "中学",
    "初中",
    "高中",
    "secondary",
    "middle school",
    "high school",
    "junior high",
    "senior high",
]

HIGHER_KEYWORDS = [
    "大学",
    "学院",
    "高等",
    "university",
    "college",
]

TYPE_SIGMA_CANDIDATES_M = [
    75.0,
    100.0,
    125.0,
    150.0,
    175.0,
    200.0,
    225.0,
    250.0,
    275.0,
    300.0,
    325.0,
    350.0,
    375.0,
    400.0,
    500.0,
    600.0,
    700.0,
    800.0,
    900.0,
    1000.0,
    1100.0,
    1200.0,
    1300.0,
    1400.0,
    1500.0,
    1600.0,
    1700.0,
    1800.0,
    1900.0,
    2000.0,
]

SCHOOL_BASE_SIGMA_M = max(
    TYPE_SIGMA_CANDIDATES_M
)

IGNORE_VALUE = 255

# 中断后重跑时，已成功生成的 cache 直接跳过。
SKIP_EXISTING = True

# 每生成多少张刷新一次 manifest。
MANIFEST_FLUSH_EVERY = 10

# GeoTIFF 可能为 EPSG:4326。
# 上海使用 UTM 51N 计算真实米/像素。
METRIC_CRS = pyproj.CRS.from_epsg(
    32651
)


# ============================================================
# 3. BASIC HELPERS
# ============================================================

def check_inputs():
    required = [
        DATASET_ROOT,
        IMAGE_DIR,
        MASK_DIR,
        METADATA_CSV,
    ]

    for p in required:
        if not p.exists():
            raise FileNotFoundError(
                "Required path not found:\n{}"
                .format(
                    p
                )
            )

    if not ALL_SCHOOL_POI_SHP.exists():
        raise FileNotFoundError(
            "Raw school POI source does not exist:\n{}"
            .format(
                ALL_SCHOOL_POI_SHP
            )
        )

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


def find_image_file(
    grid_id
):
    for suffix in [
        ".tif",
        ".tiff",
        ".TIF",
        ".TIFF",
    ]:
        p = (
            IMAGE_DIR
            / "{}{}".format(
                grid_id,
                suffix
            )
        )

        if p.exists():
            return p

    matches = [
        p
        for p in IMAGE_DIR.iterdir()
        if (
            p.is_file()
            and p.stem == grid_id
            and p.suffix.lower()
            in {
                ".tif",
                ".tiff",
            }
        )
    ]

    if len(matches) != 1:
        raise FileNotFoundError(
            "Cannot uniquely locate image for {} in {}\nmatches={}"
            .format(
                grid_id,
                IMAGE_DIR,
                matches,
            )
        )

    return matches[0]


def read_mask_shape(
    grid_id
):
    path = (
        MASK_DIR
        / "{}.png".format(
            grid_id
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with Image.open(
        str(
            path
        )
    ) as im:
        w, h = im.size

    return (
        int(h),
        int(w),
    )


# ============================================================
# 4. GEOTIFF HELPERS
# ============================================================

def tag_values(
    tags,
    name
):
    if name not in tags:
        return None

    v = tags[
        name
    ].value

    if isinstance(
        v,
        tuple
    ):
        return v

    if isinstance(
        v,
        list
    ):
        return tuple(
            v
        )

    try:
        return tuple(
            v
        )

    except TypeError:
        return (
            v,
        )


def read_geotiff_info(
    image_path
):
    """
    North-up GeoTIFF only, same assumption as the earlier POI scripts.
    """
    with tifffile.TiffFile(
        str(
            image_path
        )
    ) as tif:
        page = tif.pages[
            0
        ]

        tags = page.tags

        scale = tag_values(
            tags,
            "ModelPixelScaleTag"
        )

        tie = tag_values(
            tags,
            "ModelTiepointTag"
        )

        keys = tag_values(
            tags,
            "GeoKeyDirectoryTag"
        )

        width = int(
            page.imagewidth
        )
        height = int(
            page.imagelength
        )

    if (
        scale is None
        or tie is None
    ):
        raise RuntimeError(
            "GeoTIFF georeferencing tags are missing:\n{}"
            .format(
                image_path
            )
        )

    epsg = None

    if (
        keys is not None
        and len(
            keys
        ) >= 4
    ):
        nkeys = int(
            keys[
                3
            ]
        )

        for i in range(
            nkeys
        ):
            b = (
                4
                + 4 * i
            )

            if (
                b + 3
                >= len(
                    keys
                )
            ):
                break

            key_id = int(
                keys[
                    b
                ]
            )

            location = int(
                keys[
                    b + 1
                ]
            )

            count = int(
                keys[
                    b + 2
                ]
            )

            value = int(
                keys[
                    b + 3
                ]
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
            "Could not determine EPSG from GeoTIFF:\n{}"
            .format(
                image_path
            )
        )

    return {
        "width": width,
        "height": height,
        "scale_x": float(
            scale[
                0
            ]
        ),
        "scale_y": float(
            scale[
                1
            ]
        ),
        "tie_i": float(
            tie[
                0
            ]
        ),
        "tie_j": float(
            tie[
                1
            ]
        ),
        "tie_x": float(
            tie[
                3
            ]
        ),
        "tie_y": float(
            tie[
                4
            ]
        ),
        "epsg": int(
            epsg
        ),
    }


def map_to_pixel(
    x,
    y,
    geo
):
    col = (
        geo[
            "tie_i"
        ]
        + (
            float(
                x
            )
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
            - float(
                y
            )
        )
        / geo[
            "scale_y"
        ]
    )

    return (
        row,
        col,
    )


def metre_scales_for_image(
    geo
):
    """
    Convert one-pixel east/south steps at image centre to metres.
    """
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
        width
        - 1
    ) / 2.0

    center_row = (
        height
        - 1
    ) / 2.0

    x0 = (
        geo[
            "tie_x"
        ]
        + (
            center_col
            - geo[
                "tie_i"
            ]
        )
        * geo[
            "scale_x"
        ]
    )

    y0 = (
        geo[
            "tie_y"
        ]
        - (
            center_row
            - geo[
                "tie_j"
            ]
        )
        * geo[
            "scale_y"
        ]
    )

    x_east = (
        x0
        + geo[
            "scale_x"
        ]
    )

    y_south = (
        y0
        - geo[
            "scale_y"
        ]
    )

    transformer = (
        pyproj.Transformer.from_crs(
            image_crs,
            METRIC_CRS,
            always_xy=True,
        )
    )

    mx0, my0 = (
        transformer.transform(
            x0,
            y0,
        )
    )

    mxe, mye = (
        transformer.transform(
            x_east,
            y0,
        )
    )

    mxs, mys = (
        transformer.transform(
            x0,
            y_south,
        )
    )

    mpp_x = float(
        (
            (
                mxe - mx0
            ) ** 2
            + (
                mye - my0
            ) ** 2
        ) ** 0.5
    )

    mpp_y = float(
        (
            (
                mxs - mx0
            ) ** 2
            + (
                mys - my0
            ) ** 2
        ) ** 0.5
    )

    if not (
        mpp_x > 0.0
        and mpp_y > 0.0
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


# ============================================================
# 5. ALL RAW SCHOOL POI + TYPE MAPPING
# ============================================================

def load_all_school_poi():
    print(
        "Loading ALL raw school POI:"
    )
    print(
        "  {}".format(
            ALL_SCHOOL_POI_SHP
        )
    )

    poi = gpd.read_file(
        str(
            ALL_SCHOOL_POI_SHP
        )
    )

    if poi.crs is None:
        raise RuntimeError(
            "Raw school POI has no CRS."
        )

    poi = poi[
        poi.geometry.notna()
        & (~poi.geometry.is_empty)
    ].copy()

    if poi.empty:
        raise RuntimeError(
            "Raw school POI is empty."
        )

    if (
        SCHOOL_TYPE_FIELD
        not in poi.columns
    ):
        raise RuntimeError(
            "Raw school POI does not contain type field '{}'.\ncolumns={}"
            .format(
                SCHOOL_TYPE_FIELD,
                list(
                    poi.columns
                ),
            )
        )

    return poi


def normalize_text(
    value
):
    if pd.isna(
        value
    ):
        return ""

    return str(
        value
    ).strip().lower()


def classify_school_type(
    value
):
    s = normalize_text(
        value
    )

    if not s:
        return None

    for kw in PRIMARY_KEYWORDS:
        if kw.lower() in s:
            return "primary"

    for kw in MIDDLE_KEYWORDS:
        if kw.lower() in s:
            return "middle"

    for kw in HIGHER_KEYWORDS:
        if kw.lower() in s:
            return "higher"

    return None


def build_typed_poi_source(
    poi,
    image_epsg
):
    """
    From ALL raw school POIs, keep ONLY the three target school types:
        primary
        middle
        higher

    POIs whose type cannot be mapped to these three classes are NOT used in
    the spatial-prior cache. They are still written to poi_type_audit.csv for
    manual traceability.
    """
    audit = poi.copy()

    audit[
        "_poi_type"
    ] = [
        classify_school_type(
            x
        )
        for x in audit[
            SCHOOL_TYPE_FIELD
        ].tolist()
    ]

    audit[
        "_used_for_cache"
    ] = [
        int(
            x in TYPE_NAMES
        )
        for x in audit[
            "_poi_type"
        ].tolist()
    ]

    audit_no_geom = pd.DataFrame(
        audit.drop(
            columns="geometry"
        )
    )

    audit_no_geom.to_csv(
        str(
            POI_TYPE_AUDIT_CSV
        ),
        index=False,
        encoding="utf-8-sig",
    )

    ignored = audit[
        audit[
            "_used_for_cache"
        ]
        == 0
    ].copy()

    if not ignored.empty:
        ignored_values = sorted(
            {
                str(x)
                for x in ignored[
                    SCHOOL_TYPE_FIELD
                ].dropna().tolist()
            }
        )
    else:
        ignored_values = []

    usable = audit[
        audit[
            "_used_for_cache"
        ]
        == 1
    ].copy()

    if usable.empty:
        raise RuntimeError(
            "No raw school POIs remain after keeping only primary/middle/higher."
        )

    image_crs = (
        pyproj.CRS.from_epsg(
            int(
                image_epsg
            )
        )
    )

    poi_img = usable.to_crs(
        image_crs
    )

    typed_xy = {}

    for type_name in TYPE_NAMES:
        sub = poi_img[
            poi_img[
                "_poi_type"
            ]
            == type_name
        ]

        xy = [
            (
                float(
                    geom.x
                ),
                float(
                    geom.y
                ),
            )
            for geom in sub.geometry
        ]

        typed_xy[
            type_name
        ] = xy

    counts = {
        name: len(
            typed_xy[
                name
            ]
        )
        for name in TYPE_NAMES
    }

    print()
    print(
        "ALL raw school POI type counts used for cache:"
    )

    for name in TYPE_NAMES:
        print(
            "  {:>8s} ({}) : {}"
            .format(
                name,
                TYPE_LABELS_CN[
                    name
                ],
                counts[
                    name
                ],
            )
        )

    print(
        "  ignored/unmapped     : {}".format(
            len(
                ignored
            )
        )
    )

    if ignored_values:
        print(
            "  ignored original type values:"
        )

        for x in ignored_values:
            print(
                "    {}".format(
                    x
                )
            )

    return (
        typed_xy,
        counts,
        int(
            len(
                ignored
            )
        ),
        ignored_values,
    )


# ============================================================
# 6. PRIOR CONSTRUCTION
# ============================================================


def poi_pixels_for_image(
    geo,
    points_xy
):
    return [
        map_to_pixel(
            x,
            y,
            geo,
        )
        for x, y in points_xy
    ]


def filter_pixels_for_sigma(
    pixels,
    h,
    w,
    mpp_x,
    mpp_y,
    sigma_m
):
    radius_x = int(
        math.ceil(
            4.0
            * float(
                sigma_m
            )
            / float(
                mpp_x
            )
        )
    )

    radius_y = int(
        math.ceil(
            4.0
            * float(
                sigma_m
            )
            / float(
                mpp_y
            )
        )
    )

    return [
        (
            r,
            c,
        )
        for r, c in pixels
        if (
            -radius_y
            <= r
            <= (
                h
                - 1
                + radius_y
            )
            and
            -radius_x
            <= c
            <= (
                w
                - 1
                + radius_x
            )
        )
    ]


def build_poi_prior(
    height,
    width,
    poi_pixels,
    metres_per_pixel_x,
    metres_per_pixel_y,
    sigma_m
):
    """
    Exact same max-Gaussian construction used in the previous POI scripts.
    """
    if not poi_pixels:
        return np.zeros(
            (
                height,
                width,
            ),
            dtype=np.float32,
        )

    if sigma_m <= 0:
        raise RuntimeError(
            "Invalid sigma."
        )

    prior = np.zeros(
        (
            height,
            width,
        ),
        dtype=np.float32,
    )

    radius_x = int(
        math.ceil(
            4.0
            * sigma_m
            / metres_per_pixel_x
        )
    )

    radius_y = int(
        math.ceil(
            4.0
            * sigma_m
            / metres_per_pixel_y
        )
    )

    yy = np.arange(
        height,
        dtype=np.float32,
    )

    xx = np.arange(
        width,
        dtype=np.float32,
    )

    denom = (
        2.0
        * sigma_m
        * sigma_m
    )

    for r0, c0 in poi_pixels:
        rc = int(
            round(
                r0
            )
        )

        cc = int(
            round(
                c0
            )
        )

        rmin = max(
            0,
            rc - radius_y,
        )

        rmax = min(
            height,
            rc + radius_y + 1,
        )

        cmin = max(
            0,
            cc - radius_x,
        )

        cmax = min(
            width,
            cc + radius_x + 1,
        )

        if (
            rmin >= rmax
            or cmin >= cmax
        ):
            continue

        y = yy[
            rmin:rmax
        ]

        x = xx[
            cmin:cmax
        ]

        dy_m = (
            (
                y[
                    :,
                    None
                ]
                - float(
                    r0
                )
            )
            * metres_per_pixel_y
        )

        dx_m = (
            (
                x[
                    None,
                    :
                ]
                - float(
                    c0
                )
            )
            * metres_per_pixel_x
        )

        d2_m = (
            dy_m
            * dy_m
            + dx_m
            * dx_m
        )

        local = np.exp(
            -d2_m
            / denom
        ).astype(
            np.float32
        )

        prior[
            rmin:rmax,
            cmin:cmax
        ] = np.maximum(
            prior[
                rmin:rmax,
                cmin:cmax
            ],
            local,
        )

    return prior


# ============================================================
# 7. CACHE VALIDATION / SAVE
# ============================================================

def cache_path_for_gid(
    gid
):
    return (
        CACHE_DIR
        / "{}.npz".format(
            gid
        )
    )


def cache_is_valid(
    path,
    expected_h,
    expected_w
):
    if not path.exists():
        return False

    try:
        with np.load(
            str(
                path
            ),
            allow_pickle=False,
        ) as obj:
            school_base = obj[
                "school_base"
            ]

            local_counts = obj[
                "local_poi_counts"
            ]

            base_sigma = float(
                obj[
                    "base_sigma_m"
                ][
                    0
                ]
            )

            if (
                school_base.shape
                != (
                    3,
                    expected_h,
                    expected_w,
                )
            ):
                return False

            if (
                school_base.dtype
                != np.float32
            ):
                return False

            if (
                local_counts.shape
                != (
                    3,
                )
            ):
                return False

            if abs(
                base_sigma
                - SCHOOL_BASE_SIGMA_M
            ) > 1e-6:
                return False

    except Exception:
        return False

    return True


def save_tile_cache(
    path,
    school_base,
    local_counts,
    mpp_x,
    mpp_y,
    image_epsg
):
    temp = Path(
        str(
            path
        )
        + ".tmp.npz"
    )

    if temp.exists():
        temp.unlink()

    np.savez_compressed(
        str(
            temp
        ),
        school_base=school_base.astype(
            np.float32,
            copy=False,
        ),
        local_poi_counts=np.asarray(
            local_counts,
            dtype=np.int32,
        ),
        mpp_xy=np.asarray(
            [
                mpp_x,
                mpp_y,
            ],
            dtype=np.float64,
        ),
        base_sigma_m=np.asarray(
            [
                SCHOOL_BASE_SIGMA_M
            ],
            dtype=np.float64,
        ),
        image_epsg=np.asarray(
            [
                int(
                    image_epsg
                )
            ],
            dtype=np.int32,
        ),
        type_names=np.asarray(
            TYPE_NAMES,
            dtype="U16",
        ),
    )

    if path.exists():
        path.unlink()

    temp.replace(
        path
    )


# ============================================================
# 8. MAIN
# ============================================================

def main():
    print(
        "=" * 96
    )
    print(
        "ONE-TIME ALL RAW SCHOOL POI BASE-PRIOR CACHE"
    )
    print(
        "=" * 96
    )

    check_inputs()

    metadata = pd.read_csv(
        str(
            METADATA_CSV
        ),
        encoding="utf-8-sig",
    )

    if "grid_id" not in metadata.columns:
        raise RuntimeError(
            "metadata.csv has no grid_id column."
        )

    grid_ids = [
        str(
            x
        )
        for x in metadata[
            "grid_id"
        ].tolist()
    ]

    if len(
        grid_ids
    ) != len(
        set(
            grid_ids
        )
    ):
        raise RuntimeError(
            "Duplicate grid_id in metadata.csv."
        )

    if not grid_ids:
        raise RuntimeError(
            "No grid IDs found."
        )

    print(
        "Dataset tiles: {}".format(
            len(
                grid_ids
            )
        )
    )

    probe_image = find_image_file(
        grid_ids[
            0
        ]
    )

    probe_geo = read_geotiff_info(
        probe_image
    )

    image_epsg = int(
        probe_geo[
            "epsg"
        ]
    )

    print(
        "Image EPSG: {}".format(
            image_epsg
        )
    )

    poi = load_all_school_poi()

    (
        typed_xy,
        type_counts,
        ignored_type_poi_count,
        ignored_type_values,
    ) = build_typed_poi_source(
        poi,
        image_epsg,
    )

    manifest_rows = []

    done = 0
    skipped = 0
    failed = 0

    start_all = time.time()

    for i, gid in enumerate(
        grid_ids,
        1
    ):
        image_path = find_image_file(
            gid
        )

        mask_h, mask_w = read_mask_shape(
            gid
        )

        out_path = cache_path_for_gid(
            gid
        )

        if (
            SKIP_EXISTING
            and cache_is_valid(
                out_path,
                mask_h,
                mask_w,
            )
        ):
            skipped += 1

            manifest_rows.append({
                "grid_id": gid,
                "status": "skipped_existing_valid",
                "cache_path": str(
                    out_path
                ),
                "height": mask_h,
                "width": mask_w,
                "seconds": 0.0,
            })

            if (
                i % 10 == 0
                or i == len(
                    grid_ids
                )
            ):
                print(
                    "[{}/{}] {} | skip existing"
                    .format(
                        i,
                        len(
                            grid_ids
                        ),
                        gid,
                    )
                )

            continue

        tile_start = time.time()

        try:
            geo = read_geotiff_info(
                image_path
            )

            if (
                int(
                    geo[
                        "height"
                    ]
                )
                != mask_h
                or int(
                    geo[
                        "width"
                    ]
                )
                != mask_w
            ):
                raise RuntimeError(
                    "Image/mask shape mismatch for {}: "
                    "image={}x{}, mask={}x{}"
                    .format(
                        gid,
                        geo[
                            "height"
                        ],
                        geo[
                            "width"
                        ],
                        mask_h,
                        mask_w,
                    )
                )

            if (
                int(
                    geo[
                        "epsg"
                    ]
                )
                != image_epsg
            ):
                raise RuntimeError(
                    "Mixed image CRS detected for {}: {} vs probe {}"
                    .format(
                        gid,
                        geo[
                            "epsg"
                        ],
                        image_epsg,
                    )
                )

            (
                mpp_x,
                mpp_y,
            ) = metre_scales_for_image(
                geo
            )

            school_base = []

            local_counts = []

            for type_name in TYPE_NAMES:
                pixels_all = poi_pixels_for_image(
                    geo,
                    typed_xy[
                        type_name
                    ],
                )

                pixels = filter_pixels_for_sigma(
                    pixels=pixels_all,
                    h=mask_h,
                    w=mask_w,
                    mpp_x=mpp_x,
                    mpp_y=mpp_y,
                    sigma_m=SCHOOL_BASE_SIGMA_M,
                )

                prior = build_poi_prior(
                    height=mask_h,
                    width=mask_w,
                    poi_pixels=pixels,
                    metres_per_pixel_x=mpp_x,
                    metres_per_pixel_y=mpp_y,
                    sigma_m=SCHOOL_BASE_SIGMA_M,
                )

                school_base.append(
                    prior
                )

                local_counts.append(
                    int(
                        len(
                            pixels
                        )
                    )
                )

            school_base = np.stack(
                school_base,
                axis=0,
            ).astype(
                np.float32
            )

            save_tile_cache(
                path=out_path,
                school_base=school_base,
                local_counts=local_counts,
                mpp_x=mpp_x,
                mpp_y=mpp_y,
                image_epsg=image_epsg,
            )

            # Re-open once to make sure the just-written file is readable.
            if not cache_is_valid(
                out_path,
                mask_h,
                mask_w,
            ):
                raise RuntimeError(
                    "Written cache failed validation."
                )

            elapsed = (
                time.time()
                - tile_start
            )

            done += 1

            manifest_rows.append({
                "grid_id": gid,
                "status": "generated",
                "cache_path": str(
                    out_path
                ),
                "height": mask_h,
                "width": mask_w,
                "mpp_x": mpp_x,
                "mpp_y": mpp_y,
                "primary_poi_count": local_counts[
                    0
                ],
                "middle_poi_count": local_counts[
                    1
                ],
                "higher_poi_count": local_counts[
                    2
                ],
                "seconds": elapsed,
            })

            print(
                "[{}/{}] {} | P/M/H={}/{}/{} | {:.1f}s"
                .format(
                    i,
                    len(
                        grid_ids
                    ),
                    gid,
                    local_counts[
                        0
                    ],
                    local_counts[
                        1
                    ],
                    local_counts[
                        2
                    ],
                    elapsed,
                )
            )

        except Exception as e:
            failed += 1

            manifest_rows.append({
                "grid_id": gid,
                "status": "FAILED",
                "cache_path": str(
                    out_path
                ),
                "height": mask_h,
                "width": mask_w,
                "error": repr(
                    e
                ),
                "seconds": (
                    time.time()
                    - tile_start
                ),
            })

            pd.DataFrame(
                manifest_rows
            ).to_csv(
                str(
                    MANIFEST_CSV
                ),
                index=False,
                encoding="utf-8-sig",
            )

            raise

        if (
            len(
                manifest_rows
            )
            % MANIFEST_FLUSH_EVERY
            == 0
        ):
            pd.DataFrame(
                manifest_rows
            ).to_csv(
                str(
                    MANIFEST_CSV
                ),
                index=False,
                encoding="utf-8-sig",
            )

    pd.DataFrame(
        manifest_rows
    ).to_csv(
        str(
            MANIFEST_CSV
        ),
        index=False,
        encoding="utf-8-sig",
    )

    total_minutes = (
        time.time()
        - start_all
    ) / 60.0

    summary = {
        "dataset_root": str(
            DATASET_ROOT
        ),
        "school_poi_source": str(
            ALL_SCHOOL_POI_SHP
        ),
        "school_type_field": SCHOOL_TYPE_FIELD,
        "type_names": TYPE_NAMES,
        "type_counts": type_counts,
        "ignored_unmapped_raw_poi_count": int(
            ignored_type_poi_count
        ),
        "ignored_unmapped_original_type_values": (
            ignored_type_values
        ),
        "school_base_sigma_m": float(
            SCHOOL_BASE_SIGMA_M
        ),
        "supported_sigma_candidates_m": [
            float(
                x
            )
            for x in TYPE_SIGMA_CANDIDATES_M
        ],
        "cache_dtype": "float32",
        "cache_format": (
            "npz_compressed; one [3,H,W] base-prior array per tile"
        ),
        "total_tiles": int(
            len(
                grid_ids
            )
        ),
        "generated_this_run": int(
            done
        ),
        "skipped_existing": int(
            skipped
        ),
        "failed": int(
            failed
        ),
        "total_minutes_this_run": float(
            total_minutes
        ),
        "fold_independent": True,
        "note": (
            "This cache contains spatial base priors from ALL raw school POIs "
            "that can be mapped to primary/middle/higher. No valid/invalid POI "
            "screening is performed. It performs no model fitting, no split "
            "selection, and no test evaluation."
        ),
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 96
    )
    print(
        "CACHE PRECOMPUTATION FINISHED"
    )
    print(
        "=" * 96
    )
    print(
        "Generated this run : {}".format(
            done
        )
    )
    print(
        "Skipped existing   : {}".format(
            skipped
        )
    )
    print(
        "Failed             : {}".format(
            failed
        )
    )
    print(
        "Total tiles        : {}".format(
            len(
                grid_ids
            )
        )
    )
    print(
        "Time               : {:.2f} min"
        .format(
            total_minutes
        )
    )
    print(
        "Cache root:"
    )
    print(
        "  {}".format(
            CACHE_DIR
        )
    )
    print(
        "Manifest:"
    )
    print(
        "  {}".format(
            MANIFEST_CSV
        )
    )
    print()
    print(
        "后续 Fold 0~4 POI 方法可直接读取这些全部学校 POI cache，不再重新构建 Gaussian prior。"
    )


if __name__ == "__main__":
    main()
