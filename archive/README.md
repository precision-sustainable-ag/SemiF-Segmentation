# Archive

Code, configs, and docs that are no longer used by `main.py`. Nothing in `src/`
imports from here. Kept for reference; everything here is also recoverable from
the `pre-refactor` git tag (the state of `develop` before this cleanup).

| Archived path | Why |
|---|---|
| `src/viz_results.py` | Never registered as a task; reads `cfg.paths.model_dir`, which doesn't exist. |
| `src/utils/remap.py` | Never imported (`remap_masks.py` has its own remapping). |
| `src/utils/calc_mean.py` | Never imported (`data_stats.py` computes mean/std). |
| `src/utils/sql3_query.py` | Never imported (superseded by `src/query.py`). |
| `src/utils/visualize_albumentations.py` | Never imported. |
| `src/preprocessing/download_from_file.py` | Never imported. |
| `src/preprocessing/move_fullsized_data.py` | Registered as a preprocess task but never enabled in any config. |
| `src/preprocessing/helpers/get_unique_common_names.py` | Never imported. |
| `src/preprocessing/archive/` | Older query scripts, already archived in place. |
| `scripts/copy_db.sh` | Not referenced anywhere. |
| `scripts/create_training_issue.py` | Only used by the archived CI training workflow. |
| `github-workflows/` | Self-hosted CI training loop (`sync → query → preprocess → train`) against the old `agir.db`; last used May 2025. Moved out of `.github/workflows/` so it no longer runs. |
| `docs/` | Auto-generated tutorial (May 2025) describing code that has since changed. |

## SemiField auto-mask training path

Training now uses only human labels from CVAT (`mode=label`), so the path that
built training data from the SemiField pipeline's auto-generated semantic masks
in `agir.db` was archived as a unit:

| Archived path | Replaced by |
|---|---|
| `src/query.py`, `conf/query/` | `mode=label` `select` over `conf/sources` (AgIR_DB_v2, field_exploration.db). The old query targets `agir.db`'s `semif_cutouts` JSON schema, which AgIR_DB_v2 doesn't have. |
| `src/preprocessing/find_images.py`, `remove_nontargets.py`, `remap_masks.py`, `src/utils/class_groupings.py` | Human vegetation/background masks need no lookup, color-checker removal, or species remapping. |
| `src/preprocessing/grid_crop.py`, `train_val_test_split.py` | `build_dataset` (splits per image group, before tiling, kept forever) and `tile` (keeps background tiles and edges). |
| `src/inferencing/get_dataset.py` | Built inference sets with the old query. |
