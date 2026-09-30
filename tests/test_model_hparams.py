"""The model's saved hyperparameters must resolve outside mode=label, where
label.round (and the paths built from it) is unset."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")


def test_hparams_log_in_train_mode(make_cfg, tmp_path):
    from pytorch_lightning.loggers import CSVLogger

    from src.utils.model import SegmentationModule

    cfg = make_cfg("mode=train", "model.encoder_weights=null")
    model = SegmentationModule(cfg)
    assert set(model.hparams) == {"model", "train"}
    assert model.hparams["model"]["arch_name"] == cfg.model.arch_name
    logger = CSVLogger(tmp_path)
    logger.log_hyperparams(model.hparams)  # raised MissingMandatoryValue (label.round) before
    logger.save()
    assert (tmp_path / "lightning_logs/version_0/hparams.yaml").exists()

    # Checkpoints load with the caller's cfg, as train and inference do.
    path = tmp_path / "model.ckpt"
    torch.save({"state_dict": model.state_dict(), "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": "2.6.0"}, path)
    loaded = SegmentationModule.load_from_checkpoint(path, cfg=cfg, weights_only=False)
    assert loaded.cfg is cfg


def test_pick_sample_tiles_mixes_vegetation_and_spreads_over_images(tmp_path):
    import cv2
    import numpy as np

    from src.train import pick_sample_tiles

    paths = []
    for image in ("A", "B", "C"):
        for k in range(6):
            mask = np.zeros((10, 10), np.uint8)
            if k < 4:
                mask[:5] = 1  # 50% vegetation
            path = tmp_path / f"semif__{image}_y0_x{k * 10}.png"
            cv2.imwrite(str(path), mask)
            paths.append(path)
    picks = pick_sample_tiles(paths, 8, seed=0)
    assert len(picks) == len(set(picks)) == 8
    vegetated = [i for i in picks if i % 6 < 4]
    assert len(vegetated) == 6 and {paths[i].stem[7] for i in vegetated} == {"A", "B", "C"}
    assert pick_sample_tiles(paths, 8, seed=0) == picks  # reproducible
    assert len(pick_sample_tiles(paths, 100, seed=0)) == 18
