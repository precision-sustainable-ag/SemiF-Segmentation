# SemiF-Segmentation

Vegetation vs. background segmentation for SemiField and field images, trained
on human labels. Images are selected from the AgIR or field databases, moved
here with Globus, annotated as full images in CVAT, and pulled back as masks
for preprocessing and training. Each labeling round is written in
[AgIR-CVToolkit](https://github.com/precision-sustainable-ag/AgIR-CVToolkit)'s
output format.

```mermaid
flowchart LR
    DB[(AgIR_DB_v2 /<br>field_exploration.db)] -->|select| L[labels.db]
    L -->|fetch: Globus| I[full-res images]
    I -->|prepare: downscale + pre-label| C[CVAT task<br>jobs of N images]
    C -->|annotate, mark jobs completed| C
    C -->|pull_cvat| M[human masks]
    M -->|preprocess: build_dataset, tile, data_stats| T[train / val / test tiles]
    T -->|train| Model
    Model -.->|next round's pre-labels| I
```

Everything is a Hydra mode: `python main.py mode=<mode>`, with `sync`, `label`,
`preprocess`, `train`, and `inference`.

## Setup

```bash
conda env create -f environment.yaml   # or: pip install -r requirements.txt
```

Credentials live in the gitignored `.keys/keys.yaml`:

```yaml
cvat:
  url: https://app.cvat.ai     # CVAT cloud
  org_id: 123                  # every request is scoped to this organization
  username: ...
  password: ...                # or `token: ...` for a personal access token
```

Globus transfers use the `globus` CLI, as AgIR-CVToolkit's `scinet-transfer` does
(`conf/transfer/default.yaml`):
- install it and log in once: `uv tool install globus-cli`, then `globus login`;
- fill in the collection IDs under `transfer.globus.endpoints` and each root's
  `endpoint_path` in `conf/sources/default.yaml`. If a collection needs a
  data_access consent, the first transfer fails with the `globus session consent`
  command to run;
- the destination is Globus Connect Personal on this machine, which shares only
  `~/` and must be running during transfers.

Until Globus is set up, `transfer.backend=copy` copies from the NFS mounts instead.

## Labeling rounds (`mode=label`)

A round is a config file that extends `conf/label/default.yaml` (see
`conf/label/example.yaml`):

```yaml
# conf/label/r001_covercrops.yaml
defaults: [default, _self_]
round: r001_covercrops
source: field                  # field | semif  (conf/sources)
select:
  n_images: 200
  balance_by: [location]
  filters: {plant_type: COVERCROPS}   # value | [values] | "pattern%"
```

```bash
python main.py mode=label label=r001_covercrops
```

This runs five tasks, each tracked per image in the project's ledger,
`outputs/runs/<project.name>/labels.db`:

| Task | Does | Status after |
|---|---|---|
| `select` | Queries a local snapshot of the source DB, skips images already in any of the project's rounds, balance-samples up to `n_images` | `selected` |
| `fetch` | Globus transfer of the full-resolution images (one transfer per source root) | `fetched` |
| `prepare` | Downscales to `label.prepare.max_side` for CVAT; pre-labels with `label.prelabel.checkpoint` | `prepared` |
| `push_cvat` | Creates the task `<project.name>_<round>` in the CVAT project (labels `vegetation` and tag `exclude`), jobs of `cvat.segment_size` images, uploads pre-labels as editable masks | `in_cvat` |
| `pull_cvat` | Reads jobs whose state is `completed` and rasterizes their vegetation masks/polygons | `labeled` / `excluded` |

Then annotate in CVAT: fix the masks, tag unusable frames `exclude`, and mark
each job completed. Re-run the same command to pull finished jobs; it's safe to
re-run at any point, since every task only acts on images in the status it
expects. `label.retry_errors=true` retries images a task marked as `error`.

A round is one AgIR-CVToolkit run, `project.subname` being the round's name,
so its folder reads like the ones `agir-cv query` and `agir-cv scinet-transfer`
write:

```
outputs/runs/<project.name>/
├── labels.db                          # the project's ledger
└── <round>/
    ├── cfg.yaml                       # config of the latest run, with runtime (user, host, git commit, overrides) and paths
    ├── logs/<epoch>.log               # one per run; Hydra's files are in logs/hydra/
    ├── query/query.csv                # select: the round's images (image_path as the source DB stores it)
    ├── query/query_spec.json          # select: database, filters, sampling, who/when/where
    ├── globus_batch.txt               # fetch: "<source> <destination>" per file (globus_batch_<root>.txt per root if several)
    ├── transfer_manifest.json         # fetch: endpoints, destination, Globus task IDs
    ├── field-batches/... or semifield-developed-images/...   # fetch: full-resolution images, laid out as on the source storage
    ├── images/<image_id>.jpg          # prepare: the downscaled copies uploaded to CVAT
    ├── masks/<image_id>.png           # prepare: model pre-labels (0/255)
    ├── cvat_downloads/<task name>/    # pull_cvat: annotations/job_<id>.json and masks/<image_id>.png
    ├── manifest.csv                   # one row per image: status, paths, CVAT task/job/frame
    └── metrics.json                   # status counts and each task's latest summary
```

Human masks are 0/1 PNGs at the resolution they were annotated in CVAT
(`cvat_downloads/<task name>/masks/`); `tile` scales them to each image. They
aren't copied anywhere else, so keep the CVAT tasks: `pull_cvat` can always
re-export them (`label.pull_cvat.refresh=true`).

## Training

```bash
python main.py mode=preprocess   # build_dataset -> tile -> data_stats
python main.py mode=train
python main.py mode=inference
```

- `build_dataset` takes every labeled image in the project's `labels.db`
  (optionally filtered by `sources`/`rounds`) and assigns train/val/test per
  group (batch by default).
  A split is stored in `labels.db` the first time an image is seen and never
  changes, so the test set stays fixed as rounds are added.
- `tile` resizes images to `preprocess.tile.scale` and cuts `tile_size` tiles,
  keeping background tiles and image edges. `inference.rescale_factor` follows
  the same scale.
- `train` validates and tests the best checkpoint and writes `run_info.json`
  (metrics, label rounds, per-split counts) to the run's version directory.

## Configuration

| Group | What it sets |
|---|---|
| `conf/label/` | Labeling rounds: tasks, selection, downscaling, pre-label model |
| `conf/sources/` | Source DBs, snapshot policy, and the storage roots their paths resolve against |
| `conf/transfer/` | Transfer backend, globus CLI, collections, destination and transfer options |
| `conf/cvat/` | CVAT URL/organization (defaults from `.keys/keys.yaml`), project, job size, when to pull |
| `conf/preprocess/` | Split sizes and grouping, tile scale/size |
| `conf/model/`, `conf/train/`, `conf/augment/`, `conf/inference/` | Model, training, augmentation, inference |
| `conf/paths/` | Where everything is written (labeling rounds: `outputs/runs/`; preprocess, train, inference: `projects/`) |

## Tests

```bash
python -m pytest
```

## Archived code

Unused scripts, the self-hosted CI training workflows, the auto-generated
tutorial that used to live in `docs/`, and the old auto-mask training path were
moved to [archive/](archive/). See [archive/README.md](archive/README.md) for
what was moved and why.
