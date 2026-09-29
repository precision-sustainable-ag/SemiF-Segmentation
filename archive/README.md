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
