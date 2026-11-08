"""Generate school-scene maps using the existing DGTRS inference functions.

The formal pipeline calls prepare_scene_maps with its already loaded VLM module.
Standalone generation: python prepare_vlm_scene_maps.py
Read-only input check: python prepare_vlm_scene_maps.py --check-inputs
Only generated cache files are written. Images, labels and weights are read-only.
"""

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path


def dataset_ids(dataset_root):
    """Use the published test-fold union and require agreement with metadata."""
    dataset_root = Path(dataset_root)
    combined = []
    for fold_id in range(5):
        path = dataset_root / "folds" / f"fold_{fold_id}" / "test.txt"
        ids = [s.strip() for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate test IDs: {path}")
        combined.extend(ids)
    if not combined or len(combined) != len(set(combined)):
        raise ValueError("Each sample must occur in exactly one test fold.")
    for gid in combined:
        if any(s in gid for s in ("/", "\\")) or gid in {".", ".."}:
            raise ValueError(f"Invalid image ID: {gid!r}")
    with (dataset_root / "metadata.csv").open(encoding="utf-8-sig", newline="") as source:
        metadata = list(csv.DictReader(source))
    meta_ids = [row["grid_id"] for row in metadata]
    if len(meta_ids) != len(set(meta_ids)) or set(meta_ids) != set(combined):
        raise ValueError("Published test-fold IDs and metadata IDs do not match.")
    return sorted(combined)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_array(np, path, array):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            np.save(output, array, allow_pickle=False)
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _valid_maps(np, margin_path, rank_path, shape, max_rank):
    if not margin_path.is_file() or not rank_path.is_file():
        return False
    try:
        margin = np.load(margin_path, mmap_mode="r", allow_pickle=False)
        rank = np.load(rank_path, mmap_mode="r", allow_pickle=False)
        return (
            margin.shape == rank.shape == shape
            and margin.dtype == np.dtype("float32")
            and rank.dtype == np.dtype("uint8")
            and bool(np.isfinite(margin).all())
            and bool(((rank >= 1) & (rank <= max_rank)).all())
        )
    except (OSError, ValueError, EOFError):
        return False


def prepare_scene_maps(vlm_module, dataset_root, cache_dir):
    """Reuse build_text_config/load_dgtrs/infer_scene_map without running old main()."""
    vlm = vlm_module
    np = vlm.np
    dataset_root = Path(dataset_root).resolve()
    cache_dir = Path(cache_dir).resolve()
    ids = dataset_ids(dataset_root)
    vlm.DATASET_ROOT = dataset_root
    vlm.IMAGE_DIR = dataset_root / "images"
    vlm.MASK_DIR = dataset_root / "masks_png"
    text_config = vlm.build_text_config()
    shapes = {}
    image_hashes = {}
    print(f"Checking DGTRS inputs for {len(ids)} images...", flush=True)
    for gid in ids:
        path = vlm.find_file(vlm.IMAGE_DIR, gid, vlm.IMAGE_SUFFIXES)
        with vlm.Image.open(path) as image:
            shapes[gid] = (image.height, image.width)
        image_hashes[gid] = file_sha256(path)
    model_dir = Path(vlm.DGTRS_ROOT) / "model"
    required_sources = ["__init__.py", "longclip.py", "model_longclip.py", "simple_tokenizer.py", "bpe_simple_vocab_16e6.txt.gz"]
    for name in required_sources:
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)
    implementation_files = sorted(model_dir.glob("*.py")) + sorted(model_dir.glob("*.gz"))
    signature = {
        "schema": 1,
        "dataset_version": (dataset_root / "dataset_version.txt").read_text(encoding="utf-8-sig").strip(),
        "images_sha256": image_hashes,
        "checkpoint_sha256": file_sha256(vlm.DGTRS_CKPT),
        "class_file_sha256": file_sha256(vlm.LARSE_LABEL_FILE),
        "scene_implementation_sha256": file_sha256(vlm.__file__),
        "dgtrs_implementation_sha256": {p.name: file_sha256(p) for p in implementation_files},
        "text_config": text_config,
        "patch_size": int(vlm.PATCH_SIZE),
        "stride": int(vlm.STRIDE),
        "batch_size": int(vlm.VLM_BATCH_SIZE),
        "torch_version": str(vlm.torch.__version__),
        "numpy_version": str(np.__version__),
    }
    manifest_path = cache_dir / "scene_manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != signature:
            raise RuntimeError(
                "Scene-cache inputs/configuration changed. Select a new cache directory "
                "for this run; existing cache files were left unchanged."
            )
    elif cache_dir.exists() and any(cache_dir.glob("*.npy")):
        raise RuntimeError("Unversioned scene cache found. Use a new empty cache directory.")

    max_rank = len(text_config["negative_labels"]) + 1
    missing = []
    for gid in ids:
        margin_path, rank_path = vlm.scene_paths(cache_dir, gid)
        if not _valid_maps(np, margin_path, rank_path, shapes[gid], max_rank):
            missing.append(gid)
    if not missing:
        print(f"DGTRS scene maps complete: {len(ids)} images.", flush=True)
        return

    # Load before creating any output, so missing model dependencies leave no cache.
    model, preprocess, text_features, device = vlm.load_dgtrs(text_config)
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if not manifest_path.exists():
            with manifest_path.open("x", encoding="utf-8") as output:
                json.dump(signature, output, ensure_ascii=False, indent=2)
        print(f"Generating DGTRS scene maps: {len(missing)} images; device={device}", flush=True)
        for index, gid in enumerate(missing, 1):
            with vlm.read_rgb(gid) as rgb:
                margin, rank = vlm.infer_scene_map(
                    gid, rgb, model, preprocess, text_features, device, text_config
                )
            if margin.shape != shapes[gid] or rank.shape != shapes[gid]:
                raise ValueError(f"Unexpected scene-map shape: {gid}")
            if not np.isfinite(margin).all() or not ((rank >= 1) & (rank <= max_rank)).all():
                raise ValueError(f"Unexpected scene-map values: {gid}")
            margin_path, rank_path = vlm.scene_paths(cache_dir, gid)
            _atomic_array(np, margin_path, margin.astype(np.float32))
            _atomic_array(np, rank_path, rank.astype(np.uint8))
            print(f"  scene [{index}/{len(missing)}] {gid}", flush=True)
    finally:
        del model, text_features
        if vlm.torch.cuda.is_available():
            vlm.torch.cuda.empty_cache()
    print(f"DGTRS scene maps ready: {len(ids)} images under {cache_dir}", flush=True)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=root / "data")
    parser.add_argument("--cache-dir", type=Path, default=root / "work" / "vlm_scene_maps")
    parser.add_argument("--check-inputs", action="store_true")
    args = parser.parse_args()
    ids = dataset_ids(args.dataset_root)
    if args.check_inputs:
        required = [
            args.dataset_root / "dataset_version.txt",
            root / "configs" / "scene_classes.txt",
            root / "pretrained" / "dgtrs" / "DGTRS-CLIP-ViT-B-16.pt",
        ]
        required += [root / "third_party" / "DGTRS" / "model" / name for name in
                     ["__init__.py", "longclip.py", "model_longclip.py", "simple_tokenizer.py", "bpe_simple_vocab_16e6.txt.gz"]]
        image_paths = {}
        for path in (args.dataset_root / "images").iterdir():
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
                if path.stem in image_paths:
                    raise ValueError(f"Ambiguous image ID: {path.stem}")
                image_paths[path.stem] = path
        for gid in ids:
            if gid not in image_paths:
                raise FileNotFoundError(f"Image not found: {gid}")
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        print(f"PASS: {len(ids)} published IDs and required image/model/text files found.")
        print("Read-only check: no model loaded, no cache generated, no files changed.")
        return
    import train_vlm_recovery as vlm

    prepare_scene_maps(vlm, args.dataset_root, args.cache_dir)


if __name__ == "__main__":
    main()
