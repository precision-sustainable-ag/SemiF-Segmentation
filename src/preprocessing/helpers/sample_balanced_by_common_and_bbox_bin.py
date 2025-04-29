import sqlite3
import logging
import pandas as pd
from pathlib import Path
import hydra
from omegaconf import DictConfig
import json

log = logging.getLogger(__name__)

class CutoutSampler:
    def __init__(self, 
                 project_name: str, 
                 db_path: Path, 
                 output_dir: Path, 
                 table_name: str = "semif_cutouts", 
                 samples_per_group: int = 100,
                 allowed_common_names: list[str] = None
                 ):
        self.project_name = project_name
        self.db_path = db_path
        self.output_dir = output_dir
        self.table_name = table_name
        self.samples_per_group = samples_per_group
        self.allowed_common_names = [name.lower() for name in allowed_common_names] if allowed_common_names else []

    def connect_db(self):
        """Connect to the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            log.info(f"Connected to database: {self.db_path}")
            return conn
        except sqlite3.Error as e:
            log.error(f"Database connection error: {e}")
            raise

    def fetch_data(self, conn: sqlite3.Connection) -> pd.DataFrame:
        """Fetch and bin the data from the database."""
        # bin_case_stmt = """
        #     CASE
        #         WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 0 AND json_extract(cutout_props, '$.bbox_area_cm2') < 0.1 THEN 'very_small'
        #         WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 0.1 AND json_extract(cutout_props, '$.bbox_area_cm2') < 1 THEN 'small'
        #         WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 1 AND json_extract(cutout_props, '$.bbox_area_cm2') < 10 THEN 'medium'
        #         WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 10 AND json_extract(cutout_props, '$.bbox_area_cm2') < 100 THEN 'medium_large'
        #         WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 100 AND json_extract(cutout_props, '$.bbox_area_cm2') < 1000 THEN 'large'
        #         WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 1000 THEN 'very_large'
        #     END
        # """

        bin_case_stmt = """
            CASE
                WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 10 AND json_extract(cutout_props, '$.bbox_area_cm2') < 100 THEN 'medium_large'
                WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 100 AND json_extract(cutout_props, '$.bbox_area_cm2') < 1000 THEN 'large'
                WHEN json_extract(cutout_props, '$.bbox_area_cm2') >= 1000 THEN 'very_large'
            END
        """

        query = f"""
            SELECT *, 
                LOWER(TRIM(json_extract(category, '$.common_name'))) AS common_name,
                {bin_case_stmt} AS bbox_bin
            FROM {self.table_name}
            WHERE json_extract(category, '$.common_name') IS NOT NULL
              AND json_extract(cutout_props, '$.bbox_area_cm2') IS NOT NULL
        """
        df = pd.read_sql_query(query, conn)

        log.info(f"Fetched {len(df)} rows from the database.")

        if self.allowed_common_names:
            df = df[df["common_name"].isin(self.allowed_common_names)]
            log.info(f"Filtered to {len(df)} rows matching allowed common names.")
        return df

    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample the data grouped by common name and bounding box bin."""
        df = df.dropna(subset=["common_name", "bbox_bin"])
        sampled_df = df.groupby(["common_name", "bbox_bin"]).apply(
            lambda g: g.sample(min(len(g), self.samples_per_group), random_state=42)
        ).reset_index(drop=True)
        log.info(f"Sampled dataframe shape: {sampled_df.shape}")
        return sampled_df
    
    def parse_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse the stringified JSON columns into nested dictionaries."""
        for col in ["cutout_props", "category"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        log.info(f"Parsed 'cutout_props' and 'category' columns into dictionaries.")
        return df

    def save_samples(self, df: pd.DataFrame):
        """Save the sampled dataframe to a JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_file = self.output_dir / f"{self.project_name}.json"
        df = self.parse_json_columns(df)
        df.to_json(output_file, orient="records", indent=4)
        log.info(f"Saved sampled data to {output_file}")

    def run(self):
        """Full pipeline: connect, fetch, sample, and save."""
        conn = self.connect_db()
        try:
            df = self.fetch_data(conn)
            sampled_df = self.sample_data(df)
            self.save_samples(sampled_df)
        finally:
            conn.close()
            log.info("Database connection closed.")


@hydra.main(version_base="1.2", config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    samples_per_group=100
    
    sampler = CutoutSampler(
        project_name=cfg.project.name,
        db_path=Path(cfg.paths.db.db_file),
        output_dir=Path(cfg.paths.project_query_dir),
        samples_per_group=samples_per_group,
        allowed_common_names=cfg.queries.category.common_name
    )
    sampler.run()


if __name__ == "__main__":
    main()
