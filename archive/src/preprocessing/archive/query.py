import sqlite3
import logging
import json
from pathlib import Path

import pandas as pd
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

class CutoutSampler:
    def __init__(self, cfg: DictConfig, table_name: str = "semif_cutouts"):
        self.cfg = cfg
        self.project_name=cfg.project.name
        self.db_path=Path(cfg.paths.db.db_file)
        self.output_dir=Path(cfg.paths.project_query_dir)
        self.table_name = table_name

        self.setup_sampling_config()

    def setup_sampling_config(self):
        """Set up the sampling configuration."""
        sample_cfg = self.cfg.query.sample

        self.random_state = sample_cfg.random_state
        
        self.domains = None
        self.samples_per_unique_domain = None
        self.dataset_size = None
        self.strategy = None
        self.replace = None

        if sample_cfg.multi_domain_balanced.enabled:
            self.strategy = "balanced"
            self.domains = sample_cfg.multi_domain_balanced.domains
            self.samples_per_unique_domain = sample_cfg.multi_domain_balanced.samples_per_unique_domain
            self.replace = sample_cfg.multi_domain_balanced.replace

        elif sample_cfg.random.enabled:
            self.strategy = "random"
            self.dataset_size = sample_cfg.random.dataset_size
            self.replace = sample_cfg.random.replace

    def connect_db(self):
        """Connect to the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            log.info(f"Connected to database: {self.db_path}")
            return conn
        except sqlite3.Error as e:
            log.error(f"Database connection error: {e}")
            raise

    def build_query_conditions(self, query_config: dict) -> str:
        conditions = []
        # category.common_name
        if query_config.category.enabled:
            log.info("Filtering by common names.")
            conditions.append("json_extract(category, '$.common_name') IS NOT NULL")
            common_names = []
            for name in query_config.category.common_name:
                safe_name = name.lower().replace("'", "''")  # Escape single quotes for SQL
                common_names.append(f"'{safe_name}'")
            conditions.append(f"LOWER(TRIM(json_extract(category, '$.common_name'))) IN ({','.join(common_names)})")

        # morphological.bbox_area_cm2
        if query_config.morphological.bbox_area_cm2.enabled:
            log.info("Filtering by bbox_area_cm2.")
            conditions.append("json_extract(cutout_props, '$.bbox_area_cm2') IS NOT NULL")
            bbox_range = query_config.morphological.bbox_area_cm2.default
            min_val = bbox_range.min
            max_val = bbox_range.max
            if min_val is not None:
                conditions.append(f"json_extract(cutout_props, '$.bbox_area_cm2') >= {min_val}")
            if max_val is not None:
                conditions.append(f"json_extract(cutout_props, '$.bbox_area_cm2') <= {max_val}")

        # morphological.blur_effect
        if query_config.morphological.blur_effect.enabled:
            log.info("Filtering by blur_effect.")
            conditions.append("json_extract(cutout_props, '$.blur_effect') IS NOT NULL")
            blur_range = query_config.morphological.blur_effect
            if blur_range.min is not None:
                conditions.append(f"json_extract(cutout_props, '$.blur_effect') >= {blur_range['min']}")
            if blur_range.max is not None:
                conditions.append(f"json_extract(cutout_props, '$.blur_effect') <= {blur_range['max']}")

        # morphological.is_primary
        if query_config.morphological.is_primary.enabled:
            log.info("Filtering by is_primary.")
            conditions.append("json_extract(cutout_props, '$.is_primary') IS NOT NULL")
            is_primary = query_config.morphological.is_primary.status
            if is_primary is not None:
                conditions.append(f"json_extract(cutout_props, '$.is_primary') = {int(is_primary)}")

        # morphological.extends_border
        if query_config.morphological.extends_border.enabled:
            log.info("Filtering by extends_border.")
            conditions.append("json_extract(cutout_props, '$.extends_border') IS NOT NULL")
            extends_border = query_config.morphological.extends_border.status
            if extends_border is not None:
                conditions.append(f"json_extract(cutout_props, '$.extends_border') = {int(extends_border)}")

        # morphological.num_components
        if query_config.morphological.num_components.enabled:
            log.info("Filtering by num_components.")
            conditions.append("json_extract(cutout_props, '$.num_components') IS NOT NULL")
            num_components = query_config.morphological.num_components
            if num_components.min is not None:
                conditions.append(f"json_extract(cutout_props, '$.num_components') >= {num_components['min']}")
            if num_components.max is not None:
                conditions.append(f"json_extract(cutout_props, '$.num_components') <= {num_components['max']}")

        # morphological.non_target_weeds
        if query_config.morphological.non_target_weed.enabled:
            log.info("Filtering by non_target_weeds.")
            non_target_weeds = query_config.morphological.non_target_weed.status
            if non_target_weeds is not None:
                conditions.append(f"json_extract(cutout_props, '$.non_target_weeds') = {int(non_target_weeds)}")

        return " AND ".join(conditions)

    def fetch_data(self, conn: sqlite3.Connection, query_config: dict) -> pd.DataFrame:
        """Fetch dynamically filtered data from the database."""
        where_clause = self.build_query_conditions(query_config)

        query = f"""
            SELECT *, 
                LOWER(TRIM(json_extract(category, '$.common_name'))) AS common_name
            FROM {self.table_name}
            WHERE {where_clause}
        """
        df = pd.read_sql_query(query, conn)
        log.info(f"Fetched {len(df)} rows from the database.")
        return df
    
    def sample_randomly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Randomly sample the dataframe."""
        if self.dataset_size is None:
            raise ValueError("dataset_size must be set for random sampling.")
        
        sampled_df = df.sample(
            n=min(len(df), self.dataset_size),
            random_state=self.random_state,
            replace=self.replace
        ).reset_index(drop=True)
        
        log.info(f"Randomly sampled {len(sampled_df)} samples.")
        return sampled_df
    
    def sample_multi_domain_balanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sample the dataframe in a balanced way across multiple domains."""
        if self.domains is None or self.samples_per_unique_domain is None:
            raise ValueError("domains and samples_per_unique_domain must be set for balanced sampling.")
        
        df = df.dropna(subset=self.domains)

        log.info(f"Balanced sampling using groupby fields: {self.domains}")
        grouped = df.groupby(list(self.domains), group_keys=False)

        # find the smallest group size
        min_group_size = min(grouped.size())
        sampled_df = df.groupby(list(self.domains), group_keys=False).sample(
            n=min(min_group_size, self.samples_per_unique_domain),
            random_state=self.random_state,
            replace=self.replace  # change to False if you do NOT want duplication
        ).reset_index(drop=True)

        log.info(f"Sampled dataframe shape: {sampled_df.shape}")
        return sampled_df
    
    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sample the data according to the specified strategy.
        - strategy='balanced': balanced across groupby fields (default)
        - strategy='random': random sampling
        """
        if self.strategy not in ["balanced", "random"]:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")

        if self.strategy == "random":
            # Random sampling
            sampled_df = self.sample_randomly(df)
        
        elif self.strategy == "balanced":
            # Balanced sampling across groupby fields
            sampled_df = self.sample_multi_domain_balanced(df)
        
        return sampled_df
    
    def add_bbox_bin_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a bbox_bin column based on cutout_props_bbox_area_cm2 value."""
        def classify_bbox(area_cm2):
            if pd.isna(area_cm2):
                return None
            if area_cm2 < 0.1:
                return "very_small"
            elif area_cm2 < 1:
                return "small"
            elif area_cm2 < 10:
                return "medium"
            elif area_cm2 < 100:
                return "medium_large"
            elif area_cm2 < 1000:
                return "large"
            else:
                return "very_large"

        if "cutout_props.bbox_area_cm2" not in df.columns:
            raise ValueError("cutout_props.bbox_area_cm2 column not found after unnesting.")

        df["bbox_bin"] = df["cutout_props.bbox_area_cm2"].apply(classify_bbox)
        log.info("Added 'bbox_bin' column based on bbox area.")
        return df
    
    def unnest_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Unnest 'cutout_props' and 'category' dictionary columns into flat DataFrame columns."""
        for col in ["cutout_props", "category"]:
            if col in df.columns:
                expanded = pd.json_normalize(df[col])
                expanded.columns = [f"{col}.{subcol}" for subcol in expanded.columns]
                df = df.drop(columns=[col]).join(expanded)
                log.info(f"Expanded '{col}' into {len(expanded.columns)} columns.")
        return df
    
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

    def run(self, query_config: dict):
        """Full pipeline: connect, fetch, sample, and save."""
        conn = self.connect_db()
        try:
            df = self.fetch_data(conn, query_config)
            df = self.parse_json_columns(df)
            df = self.unnest_json_columns(df)
            df = self.add_bbox_bin_column(df)
            sampled_df = self.sample_data(df)
            self.save_samples(sampled_df)
        finally:
            conn.close()
            log.info("Database connection closed.")

@hydra.main(version_base="1.2", config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    
    sampler = CutoutSampler(cfg)
    sampler.run(query_config=cfg.query)


if __name__ == "__main__":
    main()
