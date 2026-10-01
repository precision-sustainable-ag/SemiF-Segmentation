"""build_dataset: the training set is every human-labeled image in the
project's labels.db (outputs/runs/<project.name>/labels.db).

Each image gets a train/val/test split the first time it's seen here, and
keeps it forever (it's stored in labels.db), so the test set never changes as
rounds are added. Splits are assigned per group (by default the image's batch,
so near-duplicate frames from one collection day never straddle splits):
existing groups keep their split, and each new group goes to the split that's
furthest below its target share. Writes manifest.csv for the tile task.
"""

import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from src.labeling.ledger import LabelLedger

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")


def group_key(item: dict, group_by: str) -> str:
    if group_by == "image":
        return f"{item['source']}:{item['image_id']}"
    if group_by in ("batch", "location"):
        return f"{item['source']}:{item[group_by] or item['image_id']}"
    raise ValueError(f"preprocess.build_dataset.group_by must be image, batch, or location; got {group_by!r}")


def assign_splits(items: list[dict], targets: dict[str, float], group_by: str) -> dict[tuple[str, str], str]:
    """{(source, image_id): split} for items without one; existing splits are never changed."""
    group_split: dict[str, str] = {}
    counts: Counter = Counter()
    for item in items:
        if item["split"]:
            counts[item["split"]] += 1
            group_split.setdefault(group_key(item, group_by), item["split"])

    new_groups: dict[str, list[dict]] = {}
    for item in items:
        if not item["split"]:
            new_groups.setdefault(group_key(item, group_by), []).append(item)

    total = len(items)
    assignments: dict[tuple[str, str], str] = {}
    # Largest groups first so the small ones can even out the proportions.
    for key in sorted(new_groups, key=lambda k: (-len(new_groups[k]), k)):
        members = new_groups[key]
        split = group_split.get(key)
        if split is None:
            split = max(SPLITS, key=lambda s: (targets[s] * total - counts[s], -SPLITS.index(s)))
            group_split[key] = split
        for item in members:
            assignments[(item["source"], item["image_id"])] = split
        counts[split] += len(members)
    return assignments


def main(cfg: DictConfig) -> None:
    bcfg = cfg.preprocess.build_dataset
    targets = {"val": float(bcfg.val_size), "test": float(bcfg.test_size)}
    targets["train"] = 1.0 - targets["val"] - targets["test"]
    if targets["train"] <= 0:
        raise ValueError("preprocess.build_dataset val_size + test_size must be < 1")

    with LabelLedger(cfg.paths.labels_db) as ledger:
        items = ledger.items(status="labeled")
        if bcfg.sources:
            items = [i for i in items if i["source"] in set(bcfg.sources)]
        if bcfg.rounds:
            items = [i for i in items if i["round"] in set(bcfg.rounds)]
        if not items:
            raise RuntimeError(f"No labeled images in {cfg.paths.labels_db}; run mode=label first")

        assignments = assign_splits(items, targets, bcfg.group_by)
        for (source, image_id), split in assignments.items():
            ledger.update(source, image_id, split=split)
        for item in items:
            item["split"] = item["split"] or assignments[(item["source"], item["image_id"])]

    manifest = pd.DataFrame([
        {
            "source": i["source"], "image_id": i["image_id"], "round": i["round"], "split": i["split"],
            "group": group_key(i, bcfg.group_by), "image_path": i["native_path"], "mask_path": i["mask_path"],
        }
        for i in items
    ])
    missing = manifest[[not (Path(p).exists() and Path(m).exists())
                        for p, m in zip(manifest["image_path"], manifest["mask_path"])]]
    if len(missing):
        log.warning("Dropping %d labeled images whose image or mask file is missing: %s",
                    len(missing), list(missing["image_id"][:10]))
        manifest = manifest.drop(missing.index)

    manifest_path = Path(cfg.paths.dataset_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    by_split = manifest.groupby("split").agg(images=("image_id", "size"), groups=("group", "nunique"))
    log.info("Wrote %s (%d images, %d newly assigned a split):\n%s",
             manifest_path, len(manifest), len(assignments), by_split.to_string())
    for split in ("val", "test"):
        if split not in by_split.index:
            log.warning("No %s images: there are too few %s groups (group_by=%s) to fill every split. "
                        "Label more, or set preprocess.build_dataset.group_by=image.",
                        split, bcfg.group_by, bcfg.group_by)
