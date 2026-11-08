# -*- coding: utf-8 -*-
r"""
不同类型学校 POI 空间影响范围统计
================================

目的
----
先不与 SegFormer 融合，只回答一个独立问题：

    小学 / 中学 / 高等教育 POI
    在当前学校建筑样本中，通常需要覆盖多远才能覆盖对应学校建筑？

定义
----
以“建筑”为统计单位。

对每栋已确定类型的学校建筑：
1. 找到所有“同类型 POI”；
2. 在建筑内部按规则网格采样（默认 2 m）；
3. 对每个建筑内部采样点，计算到最近同类型 POI 的距离；
4. 将这些距离的 P95 定义为该建筑的 POI 所需覆盖半径 R95。

因此：
    R95 = 某栋建筑内部约 95% 面积对应的位置，
          到最近同类型 POI 的距离不超过该值。

然后分别对小学 / 中学 / 高等教育的所有建筑 R95 做统计，
输出：
- 每栋建筑的 R95 / P50 / P75 / P90 / max 等；
- 各类别 R95 的 P50 / P75 / P90 / P95 / P99；
- 排序曲线；
- 直方图；
- CDF 累计覆盖曲线。

重要
----
- 使用“全部原始学校 POI”，不做有效/无效 POI 筛选。
- 以建筑为统计对象，所以“周围没有建筑的 POI”天然不会形成一个统计样本，
  不需要人为删除。
- 建筑类型沿用当前样本整理规则：
      人工 type 优先；
      type 为空 -> POI-AOI 推断。
- 只统计当前 school_cores_merged.shp 核心区内的学校建筑；
  同一建筑只统计一次。
- 本脚本只是空间统计，不读取 SegFormer，不训练模型，不搜索融合参数。
- 当前默认用全部样本做“描述性分析”。
  如果后续把某个分位数直接作为正式模型参数，应在每个 Fold 的 TRAIN
  样本上重新估计，避免 test 信息进入参数确定过程。

依赖
----
numpy
pandas
geopandas
shapely
scipy
matplotlib
pyproj
"""

from __future__ import print_function

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from scipy.spatial import cKDTree
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# 1. PATHS
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent / "data" / "auxiliary"

BUILDING_SHP = (
    BASE_DIR
    / "school_buildings_merged.shp"
)

CORE_SHP = (
    BASE_DIR
    / "school_cores_merged.shp"
)

AOI_SHP = (
    BASE_DIR
    / "学校AOI.shp"
)

POI_SHP = (
    BASE_DIR
    / "学校POI.shp"
)

OUT_DIR = Path(__file__).resolve().parent / "work" / "poi_distances"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 2. SETTINGS
# =============================================================================

TYPE_FIELD = "type"
POI_MIDDLE_FIELD = "中类"

SCHOOL_TYPES = [
    "小学",
    "中学",
    "高等教育",
]

# 上海米制坐标系。
METRIC_EPSG = 32651

# 建筑内部规则采样间距。
# 2 m 对建筑尺度统计已经比较细，同时计算量不大。
SAMPLE_STEP_M = 2.0

# 每栋建筑的“影响范围”定义：
# 建筑内部采样点到最近同类 POI 距离的 P95。
BUILDING_COVERAGE_PERCENTILE = 95.0

# 直方图箱宽。
HIST_BIN_WIDTH_M = 25.0

# 图片分辨率。
FIG_DPI = 300


# =============================================================================
# 3. BASIC HELPERS
# =============================================================================

def check_inputs():
    for p in [
        BUILDING_SHP,
        CORE_SHP,
        AOI_SHP,
        POI_SHP,
    ]:
        if not p.exists():
            raise FileNotFoundError(
                "Required file not found:\n{}".format(p)
            )


def clean_gdf(gdf, name):
    if gdf.crs is None:
        raise RuntimeError(
            "{} has no CRS.".format(name)
        )

    gdf = gdf[
        gdf.geometry.notna()
        & (~gdf.geometry.is_empty)
    ].copy()

    if gdf.empty:
        raise RuntimeError(
            "{} is empty.".format(name)
        )

    bad = ~gdf.geometry.is_valid

    if bool(bad.any()):
        print(
            "{}: repairing {} invalid geometries..."
            .format(
                name,
                int(bad.sum()),
            )
        )

        gdf.loc[
            bad,
            "geometry"
        ] = (
            gdf.loc[
                bad
            ]
            .geometry
            .buffer(0)
        )

        gdf = gdf[
            gdf.geometry.notna()
            & (~gdf.geometry.is_empty)
        ].copy()

    return gdf


def union_all_compat(geo_series):
    try:
        return geo_series.union_all()
    except Exception:
        return geo_series.unary_union


def subset_intersects(gdf, geom):
    if (
        gdf is None
        or gdf.empty
        or geom is None
        or geom.is_empty
    ):
        return gdf.iloc[0:0].copy()

    try:
        idx = list(
            gdf.sindex.query(
                geom,
                predicate="intersects",
            )
        )
        result = gdf.iloc[idx].copy()

    except Exception:
        result = gdf[
            gdf.intersects(geom)
        ].copy()

    if result.empty:
        return result

    return result[
        result.intersects(geom)
    ].copy()


def locate_type_field(buildings):
    if TYPE_FIELD in buildings.columns:
        return TYPE_FIELD

    matches = [
        c
        for c in buildings.columns
        if str(c).lower()
        == TYPE_FIELD.lower()
    ]

    if matches:
        return matches[0]

    raise KeyError(
        "建筑样本中没有找到字段 [{}].\n实际字段: {}"
        .format(
            TYPE_FIELD,
            list(buildings.columns),
        )
    )


# =============================================================================
# 4. TYPE NORMALIZATION
# =============================================================================

def normalize_school_type(value):
    """
    与当前学校样本整理规则保持一致：
    小学 / 中学 / 高等教育。
    """
    if value is None or pd.isna(value):
        return None

    s = str(value).strip().lower()

    if (
        not s
        or s in {
            "nan",
            "none",
            "null",
        }
    ):
        return None

    # 小学优先，避免“大学附属小学”误判为高等教育。
    primary_words = [
        "小学",
        "primary school",
        "elementary school",
    ]

    if any(
        k in s
        for k in primary_words
    ):
        return "小学"

    secondary_words = [
        "中学",
        "初中",
        "高中",
        "初级中学",
        "高级中学",
        "middle school",
        "secondary school",
        "high school",
    ]

    if any(
        k in s
        for k in secondary_words
    ):
        return "中学"

    higher_words = [
        "高等教育",
        "大学",
        "学院",
        "高校",
        "高等院校",
        "university",
        "college",
        "higher education",
    ]

    if any(
        k in s
        for k in higher_words
    ):
        return "高等教育"

    return None


# =============================================================================
# 5. AOI + BUILDING FINAL TYPE
# =============================================================================

def classify_aoi_from_poi(
    aoi_gdf,
    poi_gdf,
):
    """
    AOI 类型由其内部全部原始学校 POI 的“中类”组合确定。
    """
    aoi = aoi_gdf.copy()

    poi = poi_gdf.to_crs(
        aoi.crs
    ).copy()

    if POI_MIDDLE_FIELD not in poi.columns:
        raise KeyError(
            "POI 中没有字段 [{}].\n实际字段: {}"
            .format(
                POI_MIDDLE_FIELD,
                list(poi.columns),
            )
        )

    poi[
        "_poi_type"
    ] = poi[
        POI_MIDDLE_FIELD
    ].apply(
        normalize_school_type
    )

    aoi_types = []

    for _, row in aoi.iterrows():
        hits = subset_intersects(
            poi,
            row.geometry,
        )

        types = {
            x
            for x in hits[
                "_poi_type"
            ].dropna().tolist()
            if x in SCHOOL_TYPES
        }

        if len(types) == 0:
            aoi_types.append(
                "不确定"
            )

        elif len(types) == 1:
            aoi_types.append(
                next(iter(types))
            )

        else:
            aoi_types.append(
                "混合"
            )

    aoi[
        "_aoi_type"
    ] = aoi_types

    return aoi


def classify_buildings(
    buildings,
    typed_aoi,
):
    """
    与当前样本整理规则保持一致：
    1) 人工 type 优先；
    2) type 为空 -> POI-AOI；
    3) AOI 归属先用 representative_point，
       找不到再用 polygon intersects。
    """
    result = (
        buildings
        .reset_index(drop=False)
        .rename(
            columns={
                "index": "_building_id"
            }
        )
        .copy()
    )

    type_field = locate_type_field(
        result
    )

    result[
        "_manual_type"
    ] = result[
        type_field
    ].apply(
        normalize_school_type
    )

    aoi = typed_aoi.to_crs(
        result.crs
    )

    final_types = []
    type_sources = []

    for _, row in result.iterrows():
        manual = row[
            "_manual_type"
        ]

        if manual in SCHOOL_TYPES:
            final_types.append(
                manual
            )
            type_sources.append(
                "manual_type"
            )
            continue

        geom = row.geometry

        if (
            geom is None
            or geom.is_empty
        ):
            final_types.append(
                None
            )
            type_sources.append(
                "unresolved"
            )
            continue

        point = geom.representative_point()

        try:
            idx = list(
                aoi.sindex.query(
                    point,
                    predicate="intersects",
                )
            )

            candidates = aoi.iloc[
                idx
            ].copy()

        except Exception:
            candidates = aoi[
                aoi.intersects(
                    point
                )
            ].copy()

        if candidates.empty:
            candidates = subset_intersects(
                aoi,
                geom,
            )

        if candidates.empty:
            final_types.append(
                None
            )
            type_sources.append(
                "unresolved"
            )
            continue

        explicit = {
            x
            for x in candidates[
                "_aoi_type"
            ].tolist()
            if x in SCHOOL_TYPES
        }

        has_mixed = bool(
            (
                candidates[
                    "_aoi_type"
                ]
                == "混合"
            ).any()
        )

        has_unknown = bool(
            (
                candidates[
                    "_aoi_type"
                ]
                == "不确定"
            ).any()
        )

        if (
            len(explicit) == 1
            and not has_mixed
            and not has_unknown
        ):
            final_types.append(
                next(iter(explicit))
            )
            type_sources.append(
                "POI-AOI"
            )

        else:
            final_types.append(
                None
            )
            type_sources.append(
                "unresolved"
            )

    result[
        "_final_type"
    ] = final_types

    result[
        "_type_source"
    ] = type_sources

    return result


# =============================================================================
# 6. BUILDING-INTERIOR SAMPLING
# =============================================================================

def sample_points_inside_geometry(
    geom,
    step_m,
):
    """
    用规则网格的像元中心近似建筑内部面积采样。

    返回:
        xy: [N,2]，单位 m

    对极小建筑，如果 2 m 网格没有落入内部，
    自动使用 representative_point 作为至少一个样本点。
    """
    if (
        geom is None
        or geom.is_empty
    ):
        return np.empty(
            (0, 2),
            dtype=np.float64,
        )

    minx, miny, maxx, maxy = (
        geom.bounds
    )

    xs = np.arange(
        minx + step_m / 2.0,
        maxx,
        step_m,
        dtype=np.float64,
    )

    ys = np.arange(
        miny + step_m / 2.0,
        maxy,
        step_m,
        dtype=np.float64,
    )

    points = []

    if (
        len(xs) > 0
        and len(ys) > 0
    ):
        # Shapely 2.x: fast vectorized point-in-polygon.
        used_vectorized = False

        try:
            from shapely import contains_xy

            xx, yy = np.meshgrid(
                xs,
                ys,
            )

            flat_x = xx.ravel()
            flat_y = yy.ravel()

            mask = contains_xy(
                geom,
                flat_x,
                flat_y,
            )

            if np.any(mask):
                points = np.column_stack(
                    [
                        flat_x[
                            mask
                        ],
                        flat_y[
                            mask
                        ],
                    ]
                ).astype(
                    np.float64
                )

            else:
                points = np.empty(
                    (0, 2),
                    dtype=np.float64,
                )

            used_vectorized = True

        except Exception:
            used_vectorized = False

        if not used_vectorized:
            # Compatibility fallback for older Shapely.
            temp = []

            for y in ys:
                for x in xs:
                    p = Point(
                        float(x),
                        float(y),
                    )

                    if geom.covers(p):
                        temp.append(
                            (
                                float(x),
                                float(y),
                            )
                        )

            if temp:
                points = np.asarray(
                    temp,
                    dtype=np.float64,
                )

            else:
                points = np.empty(
                    (0, 2),
                    dtype=np.float64,
                )

    else:
        points = np.empty(
            (0, 2),
            dtype=np.float64,
        )

    if len(points) == 0:
        rp = geom.representative_point()

        points = np.asarray(
            [
                [
                    float(rp.x),
                    float(rp.y),
                ]
            ],
            dtype=np.float64,
        )

    return points


# =============================================================================
# 7. MAIN DISTANCE ANALYSIS
# =============================================================================

def analyze_buildings(
    buildings_m,
    poi_m,
):
    """
    对每栋建筑计算到最近同类型 POI 的距离分布。
    """
    trees = {}
    poi_counts = {}

    for school_type in SCHOOL_TYPES:
        sub = poi_m[
            poi_m[
                "_poi_type"
            ] == school_type
        ].copy()

        if sub.empty:
            raise RuntimeError(
                "没有可识别的 {} POI."
                .format(
                    school_type
                )
            )

        xy = np.column_stack(
            [
                sub.geometry.x.to_numpy(
                    dtype=np.float64
                ),
                sub.geometry.y.to_numpy(
                    dtype=np.float64
                ),
            ]
        )

        trees[
            school_type
        ] = cKDTree(
            xy
        )

        poi_counts[
            school_type
        ] = int(
            len(sub)
        )

    rows = []

    total = len(
        buildings_m
    )

    for n, (_, row) in enumerate(
        buildings_m.iterrows(),
        start=1,
    ):
        school_type = row[
            "_final_type"
        ]

        if school_type not in SCHOOL_TYPES:
            continue

        geom = row.geometry

        points_xy = (
            sample_points_inside_geometry(
                geom,
                SAMPLE_STEP_M,
            )
        )

        distances, _ = trees[
            school_type
        ].query(
            points_xy,
            k=1,
            workers=-1,
        )

        distances = np.asarray(
            distances,
            dtype=np.float64,
        )

        rp = (
            geom
            .representative_point()
        )

        rp_dist, _ = trees[
            school_type
        ].query(
            np.asarray(
                [
                    [
                        float(rp.x),
                        float(rp.y),
                    ]
                ],
                dtype=np.float64,
            ),
            k=1,
        )

        rp_dist = float(
            np.asarray(
                rp_dist
            ).ravel()[0]
        )

        centroid = geom.centroid

        centroid_dist, _ = trees[
            school_type
        ].query(
            np.asarray(
                [
                    [
                        float(centroid.x),
                        float(centroid.y),
                    ]
                ],
                dtype=np.float64,
            ),
            k=1,
        )

        centroid_dist = float(
            np.asarray(
                centroid_dist
            ).ravel()[0]
        )

        rows.append({
            "building_id": int(
                row[
                    "_building_id"
                ]
            ),
            "school_type": school_type,
            "type_source": row[
                "_type_source"
            ],
            "building_area_m2": float(
                geom.area
            ),
            "n_interior_samples": int(
                len(distances)
            ),
            "sample_step_m": float(
                SAMPLE_STEP_M
            ),
            "nearest_poi_distance_representative_point_m": rp_dist,
            "nearest_poi_distance_centroid_m": centroid_dist,
            "distance_p50_m": float(
                np.percentile(
                    distances,
                    50
                )
            ),
            "distance_p75_m": float(
                np.percentile(
                    distances,
                    75
                )
            ),
            "distance_p90_m": float(
                np.percentile(
                    distances,
                    90
                )
            ),
            "distance_p95_m": float(
                np.percentile(
                    distances,
                    BUILDING_COVERAGE_PERCENTILE
                )
            ),
            "distance_p99_m": float(
                np.percentile(
                    distances,
                    99
                )
            ),
            "distance_max_m": float(
                np.max(
                    distances
                )
            ),
        })

        if (
            n % 100 == 0
            or n == total
        ):
            print(
                "  buildings [{}/{}]"
                .format(
                    n,
                    total,
                )
            )

    return (
        pd.DataFrame(
            rows
        ),
        poi_counts,
    )


# =============================================================================
# 8. SUMMARY
# =============================================================================

def make_summary(
    building_df,
    poi_counts,
):
    """
    对“每栋建筑的 R95”再做类别级统计。
    """
    rows = []

    for school_type in SCHOOL_TYPES:
        sub = building_df[
            building_df[
                "school_type"
            ] == school_type
        ].copy()

        if sub.empty:
            continue

        x = sub[
            "distance_p95_m"
        ].to_numpy(
            dtype=np.float64
        )

        rows.append({
            "school_type": school_type,
            "n_buildings": int(
                len(sub)
            ),
            "n_raw_poi_of_same_type": int(
                poi_counts[
                    school_type
                ]
            ),
            "R95_mean_m": float(
                np.mean(x)
            ),
            "R95_median_m": float(
                np.median(x)
            ),
            "R95_P75_m": float(
                np.percentile(
                    x,
                    75
                )
            ),
            "R95_P90_m": float(
                np.percentile(
                    x,
                    90
                )
            ),
            "R95_P95_m": float(
                np.percentile(
                    x,
                    95
                )
            ),
            "R95_P99_m": float(
                np.percentile(
                    x,
                    99
                )
            ),
            "R95_max_m": float(
                np.max(x)
            ),
        })

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 9. PLOTS
# =============================================================================

def save_sorted_curves(
    building_df,
):
    """
    用户提出的图：
    横轴 = 按影响范围从小到大排列的建筑样本
    纵轴 = 该建筑 R95
    """
    for school_type in SCHOOL_TYPES:
        sub = building_df[
            building_df[
                "school_type"
            ] == school_type
        ].copy()

        if sub.empty:
            continue

        x = np.sort(
            sub[
                "distance_p95_m"
            ].to_numpy(
                dtype=np.float64
            )
        )

        rank = np.arange(
            1,
            len(x) + 1,
        )

        fig, ax = plt.subplots(
            figsize=(
                7.2,
                4.8,
            )
        )

        ax.plot(
            rank,
            x,
            linewidth=1.6,
        )

        ax.set_xlabel(
            "Building sample rank"
        )

        ax.set_ylabel(
            "POI influence radius R95 (m)"
        )

        ax.set_title(
            "{}: sorted building-level POI influence radius"
            .format(
                school_type
            )
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            OUT_DIR
            / "sorted_R95_{}.png".format(
                school_type
            ),
            dpi=FIG_DPI,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )


def save_histograms(
    building_df,
):
    """
    用户提出的另一类图：
    横轴 = 影响距离
    纵轴 = 建筑样本数量
    """
    for school_type in SCHOOL_TYPES:
        sub = building_df[
            building_df[
                "school_type"
            ] == school_type
        ].copy()

        if sub.empty:
            continue

        x = sub[
            "distance_p95_m"
        ].to_numpy(
            dtype=np.float64
        )

        max_x = max(
            HIST_BIN_WIDTH_M,
            math.ceil(
                float(
                    np.max(x)
                )
                / HIST_BIN_WIDTH_M
            )
            * HIST_BIN_WIDTH_M,
        )

        bins = np.arange(
            0.0,
            max_x
            + HIST_BIN_WIDTH_M,
            HIST_BIN_WIDTH_M,
        )

        fig, ax = plt.subplots(
            figsize=(
                7.2,
                4.8,
            )
        )

        ax.hist(
            x,
            bins=bins,
        )

        ax.set_xlabel(
            "POI influence radius R95 (m)"
        )

        ax.set_ylabel(
            "Number of building samples"
        )

        ax.set_title(
            "{}: distribution of building-level POI influence radius"
            .format(
                school_type
            )
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.25,
        )

        fig.tight_layout()

        fig.savefig(
            OUT_DIR
            / "hist_R95_{}.png".format(
                school_type
            ),
            dpi=FIG_DPI,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )


def save_combined_cdf(
    building_df,
):
    """
    最推荐的图：
    横轴 = 距离
    纵轴 = R95 不超过该距离的建筑比例
    """
    fig, ax = plt.subplots(
        figsize=(
            7.4,
            5.0,
        )
    )

    plotted = False

    for school_type in SCHOOL_TYPES:
        sub = building_df[
            building_df[
                "school_type"
            ] == school_type
        ].copy()

        if sub.empty:
            continue

        x = np.sort(
            sub[
                "distance_p95_m"
            ].to_numpy(
                dtype=np.float64
            )
        )

        y = (
            np.arange(
                1,
                len(x) + 1,
                dtype=np.float64,
            )
            / float(
                len(x)
            )
            * 100.0
        )

        ax.plot(
            x,
            y,
            linewidth=1.8,
            label=school_type,
        )

        plotted = True

    if not plotted:
        plt.close(
            fig
        )
        return

    ax.set_xlabel(
        "POI influence radius R95 (m)"
    )

    ax.set_ylabel(
        "Cumulative proportion of buildings (%)"
    )

    ax.set_title(
        "Empirical distribution of POI influence radius by school type"
    )

    ax.set_ylim(
        0.0,
        100.0,
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUT_DIR
        / "CDF_R95_all_school_types.png",
        dpi=FIG_DPI,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# 10. MAIN
# =============================================================================

def main():
    print(
        "=" * 92
    )

    print(
        "SCHOOL POI INFLUENCE-RANGE ANALYSIS"
    )

    print(
        "=" * 92
    )

    check_inputs()

    print(
        "\n[1/6] Reading vectors..."
    )

    buildings = clean_gdf(
        gpd.read_file(
            BUILDING_SHP
        ),
        "buildings",
    )

    cores = clean_gdf(
        gpd.read_file(
            CORE_SHP
        ),
        "cores",
    )

    aoi = clean_gdf(
        gpd.read_file(
            AOI_SHP
        ),
        "AOI",
    )

    poi = clean_gdf(
        gpd.read_file(
            POI_SHP
        ),
        "POI",
    )

    print(
        "  buildings = {}"
        .format(
            len(buildings)
        )
    )

    print(
        "  cores     = {}"
        .format(
            len(cores)
        )
    )

    print(
        "  AOI       = {}"
        .format(
            len(aoi)
        )
    )

    print(
        "  raw POI   = {}"
        .format(
            len(poi)
        )
    )

    print(
        "\n[2/6] Determining final building types..."
    )

    typed_aoi = classify_aoi_from_poi(
        aoi,
        poi,
    )

    typed_buildings = classify_buildings(
        buildings,
        typed_aoi,
    )

    poi[
        "_poi_type"
    ] = poi[
        POI_MIDDLE_FIELD
    ].apply(
        normalize_school_type
    )

    # ---------------------------------------------------------
    # Restrict building samples to current core areas.
    # Same building is still only one vector record.
    # ---------------------------------------------------------
    cores_b = cores.to_crs(
        typed_buildings.crs
    )

    core_union = union_all_compat(
        cores_b.geometry
    )

    core_buildings = subset_intersects(
        typed_buildings,
        core_union,
    )

    core_buildings = core_buildings[
        core_buildings[
            "_final_type"
        ].isin(
            SCHOOL_TYPES
        )
    ].copy()

    unresolved_n = int(
        np.sum(
            ~typed_buildings[
                "_final_type"
            ].isin(
                SCHOOL_TYPES
            )
        )
    )

    print(
        "  usable core buildings = {}"
        .format(
            len(core_buildings)
        )
    )

    print(
        "  unresolved/non-target buildings outside analysis = {}"
        .format(
            unresolved_n
        )
    )

    print(
        "\n  Building counts:"
    )

    print(
        core_buildings[
            "_final_type"
        ]
        .value_counts()
        .reindex(
            SCHOOL_TYPES,
            fill_value=0,
        )
        .to_string()
    )

    print(
        "\n  Raw POI counts:"
    )

    print(
        poi[
            "_poi_type"
        ]
        .value_counts()
        .reindex(
            SCHOOL_TYPES,
            fill_value=0,
        )
        .to_string()
    )

    print(
        "\n[3/6] Projecting to EPSG:{}..."
        .format(
            METRIC_EPSG
        )
    )

    core_buildings_m = (
        core_buildings
        .to_crs(
            epsg=METRIC_EPSG
        )
    )

    poi_m = (
        poi[
            poi[
                "_poi_type"
            ].isin(
                SCHOOL_TYPES
            )
        ]
        .copy()
        .to_crs(
            epsg=METRIC_EPSG
        )
    )

    print(
        "\n[4/6] Computing building-level POI distance distributions..."
    )

    (
        building_df,
        poi_counts,
    ) = analyze_buildings(
        core_buildings_m,
        poi_m,
    )

    building_csv = (
        OUT_DIR
        / "building_poi_influence_R95.csv"
    )

    building_df.to_csv(
        building_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "\n[5/6] Building class-level summary..."
    )

    summary_df = make_summary(
        building_df,
        poi_counts,
    )

    summary_csv = (
        OUT_DIR
        / "school_type_R95_summary.csv"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        summary_df.to_string(
            index=False,
            float_format=lambda x: (
                "{:.2f}".format(x)
            ),
        )
    )

    print(
        "\n[6/6] Saving figures..."
    )

    save_sorted_curves(
        building_df
    )

    save_histograms(
        building_df
    )

    save_combined_cdf(
        building_df
    )

    metadata = {
        "analysis_scope": (
            "all unique school-building samples intersecting current school cores"
        ),
        "formal_model_parameter_selection": False,
        "school_poi_source": str(
            POI_SHP
        ),
        "poi_filtering": (
            "none; all raw school POIs whose middle-category maps to "
            "小学/中学/高等教育 are used"
        ),
        "building_type_rule": (
            "manual type first; empty manual type falls back to POI-AOI"
        ),
        "metric_crs_epsg": int(
            METRIC_EPSG
        ),
        "interior_sampling_step_m": float(
            SAMPLE_STEP_M
        ),
        "building_influence_radius_definition": (
            "P95 of distances from regular interior samples of each building "
            "to nearest POI of the same school type"
        ),
        "building_coverage_percentile": float(
            BUILDING_COVERAGE_PERCENTILE
        ),
        "note_for_later_modeling": (
            "If a percentile of these distances is used as a formal POI model "
            "parameter, estimate it on each fold's TRAIN subset only."
        ),
        "outputs": {
            "per_building_csv": str(
                building_csv
            ),
            "summary_csv": str(
                summary_csv
            ),
            "combined_cdf": str(
                OUT_DIR
                / "CDF_R95_all_school_types.png"
            ),
        },
    }

    with open(
        OUT_DIR
        / "analysis_definition.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "=" * 92
    )

    print(
        "DONE"
    )

    print(
        "=" * 92
    )

    print(
        "Output:"
    )

    print(
        "  {}".format(
            OUT_DIR
        )
    )

    print()
    print(
        "Most important files:"
    )

    print(
        "  {}".format(
            summary_csv
        )
    )

    print(
        "  {}".format(
            building_csv
        )
    )

    print(
        "  {}".format(
            OUT_DIR
            / "CDF_R95_all_school_types.png"
        )
    )


if __name__ == "__main__":
    main()
