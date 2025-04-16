import sqlite3
import logging
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] - %(message)s")


@hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    db_path = cfg.paths.db.db_file
    table_name = "semif_cutouts"
    samples_per_group = cfg.get("sample_per_group", 100)  # Configurable sample size

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    bin_case_stmt = """
        CASE
            WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 0 AND json_extract(cutout_props, '$.bbox_area_cm2') < 0.1 THEN 'very_small'
            WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 0.1 AND json_extract(cutout_props, '$.bbox_area_cm2') < 1 THEN 'small'
            WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 1 AND json_extract(cutout_props, '$.bbox_area_cm2') < 10 THEN 'medium'
            WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 10 AND json_extract(cutout_props, '$.bbox_area_cm2') < 100 THEN 'medium_large'
            WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 100 AND json_extract(cutout_props, '$.bbox_area_cm2') < 1000 THEN 'large'
            WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 1000 THEN 'very_large'
        END
    """

    query = f"""
        SELECT *, 
            LOWER(TRIM(json_extract(category, '$.common_name'))) AS common_name,
            {bin_case_stmt} AS bbox_bin
        FROM {table_name}
        WHERE json_extract(category, '$.common_name') IS NOT NULL
          AND json_extract(cutout_props, '$.bbox_area_cm2') IS NOT NULL
    """

    df = pd.read_sql_query(query, conn)
    conn.close()
    log.info(f"Loaded {len(df)} rows from the database.")

    # Drop rows with NULL bins or common_name
    df = df.dropna(subset=["common_name", "bbox_bin"])

    # Group and sample
    sampled_df = df.groupby(["common_name", "bbox_bin"]).apply(
        lambda g: g.sample(min(len(g), samples_per_group), random_state=42)
    ).reset_index(drop=True)

    log.info(f"Sampled dataframe shape: {sampled_df.shape}")
    
    output_path = Path(cfg.paths.data_dir) / "balanced_sample.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_path, index=False)
    log.info(f"Saved sampled data to {output_path}")

if __name__ == "__main__":
    main()
