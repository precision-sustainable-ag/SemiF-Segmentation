import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def make_cfg(tmp_path):
    """Compose the real Hydra config with everything under paths.persistent_dir
    redirected to tmp_path."""

    def _make(*overrides: str):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=str(REPO_ROOT / "conf"), version_base="1.3"):
            return compose(
                config_name="config",
                overrides=[f"paths.persistent_dir={tmp_path}", "mode=label", *overrides],
            )

    return _make
