import sqlite3
import logging
import json
from pathlib import Path
from typing import Any

import pandas as pd
import hydra
from omegaconf import DictConfig


log = logging.getLogger(__name__)

class CutoutQuerySampler:
    def __init__(self, cfg: DictConfig, table_name: str = "semif_cutouts", inference: bool = False):
        self.cfg = cfg
        self.db_path = Path(cfg.paths.db.db_file)
        self.output_dir = Path(cfg.paths.project_query_dir) if not inference else Path(cfg.paths.project_inference_dir) / "data"
        self.project_name = cfg.project.name
        self.table_name = table_name

        self.connection = None
        self.cursor = None
        self.conditions = []
        self.params = []

        self.setup_sampling_config()

    def setup_sampling_config(self):
        sample_cfg = self.cfg.query.sample
        self.random_state = sample_cfg.random_state

        self.strategy = None
        self.domains = None
        self.samples_per_unique_domain = None
        self.dataset_size = None
        self.replace = False

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
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
            log.info(f"Connected to database: {self.db_path}")
        except sqlite3.Error as e:
            log.error(f"Database connection error: {e}")
            raise

    def close_db(self):
        if self.connection:
            self.connection.commit()
            self.connection.close()
            log.info("Database connection closed.")

    def add_condition(self, column: str, operator: str, value: Any):
        condition = f"{column} {operator} ?"
        self.conditions.append(condition)
        self.params.append(value)
        log.info("Added condition: %s with value: %s", condition, value)

    def add_category_condition(self):
        category_cfg = self.cfg.query.category
        if category_cfg.enabled and category_cfg.common_name:
            names = [name.lower().strip() for name in category_cfg.common_name]
            placeholders = ", ".join("?" for _ in names)
            condition = f"LOWER(TRIM(json_extract(category, '$.common_name'))) IN ({placeholders})"
            self.conditions.append(condition)
            self.params.extend(names)

    def add_morphological_conditions(self):
        morph = self.cfg.query.morphological

        # BBox area
        if morph.bbox_area_cm2.enabled:
            default = morph.bbox_area_cm2.default
            self.add_condition("json_extract(cutout_props, '$.bbox_area_cm2')", ">=", default.min)
            self.add_condition("json_extract(cutout_props, '$.bbox_area_cm2')", "<=", default.max)

        # Blur effect
        if morph.blur_effect.enabled:
            self.add_condition("json_extract(cutout_props, '$.blur_effect')", ">=", morph.blur_effect.min)
            self.add_condition("json_extract(cutout_props, '$.blur_effect')", "<=", morph.blur_effect.max)

        # is_primary
        if morph.is_primary.enabled:
            self.add_condition("json_extract(cutout_props, '$.is_primary')", "=", int(morph.is_primary.status))

        # extends_border
        if morph.extends_border.enabled:
            self.add_condition("json_extract(cutout_props, '$.extends_border')", "=", int(morph.extends_border.status))

        # num_components
        if morph.num_components.enabled:
            self.add_condition("json_extract(cutout_props, '$.num_components')", ">=", morph.num_components.min)
            self.add_condition("json_extract(cutout_props, '$.num_components')", "<=", morph.num_components.max)

    def build_query(self) -> str:
        base_query = f"SELECT *, LOWER(TRIM(json_extract(category, '$.common_name'))) AS common_name FROM {self.table_name}"
        if self.conditions:
            where_clause = " AND ".join(self.conditions)
            query = f"{base_query} WHERE {where_clause}"
        else:
            query = base_query
        log.info(f"Built SQL Query: {query}")
        return query

    def execute_query(self) -> pd.DataFrame:
        query = self.build_query()
        log.info(f"Executing query with params: {self.params}")
        self.cursor.execute(query, self.params)
        rows = self.cursor.fetchall()
        col_names = [desc[0] for desc in self.cursor.description]
        df = pd.DataFrame(rows, columns=col_names)
        log.info(f"Query returned {len(df)} rows.")
        return df

    def parse_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["cutout_props", "category"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
        log.info("Parsed JSON columns into dicts.")
        return df

    def unnest_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in ["cutout_props", "category"]:
            if col in df.columns:
                expanded = pd.json_normalize(df[col])
                expanded.columns = [f"{col}.{subcol}" for subcol in expanded.columns]
                df = df.drop(columns=[col]).join(expanded)
                log.info(f"Expanded {col} into {len(expanded.columns)} columns.")
        return df
    
    def renest_json_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Efficiently re-nest 'cutout_props' and 'category' fields from flattened columns."""
        for prefix in ["cutout_props", "category"]:
            nested_cols = [col for col in df.columns if col.startswith(f"{prefix}.")]
            if not nested_cols:
                continue

            # Extract sub-columns
            sub_df = df[nested_cols].copy()
            sub_df.columns = [col.split(".", 1)[1] for col in nested_cols]

            # Build nested dictionaries row-wise using .to_dict
            df[prefix] = sub_df.to_dict(orient="records")

            # Drop the flattened columns
            df = df.drop(columns=nested_cols)

        log.info("Re-nested 'cutout_props' and 'category' using vectorized method.")
        return df

    def add_bbox_bin_column(self, df: pd.DataFrame) -> pd.DataFrame:
        def classify_bbox(area):
            if pd.isna(area):
                return None
            if area < 0.1:
                return "very_small"
            elif area < 1:
                return "small"
            elif area < 10:
                return "medium"
            elif area < 100:
                return "medium_large"
            elif area < 1000:
                return "large"
            else:
                return "very_large"

        if "cutout_props.bbox_area_cm2" not in df.columns:
            raise ValueError("Missing bbox area column after unnesting.")

        df["bbox_bin"] = df["cutout_props.bbox_area_cm2"].apply(classify_bbox)
        log.info("Added bbox_bin classification.")
        return df

    def sample_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.strategy == "random":
            sampled_df = df.sample(
                n=min(len(df), self.dataset_size),
                random_state=self.random_state,
                replace=self.replace
            ).reset_index(drop=True)
            log.info(f"Randomly sampled {len(sampled_df)} rows.")
            return sampled_df

        elif self.strategy == "balanced":
            grouped = df.groupby(list(self.domains), group_keys=False)
            sampled_groups = []

            for name, group in grouped:
                n = self.samples_per_unique_domain
                if not self.replace:
                    n = min(len(group), self.samples_per_unique_domain)

                sampled = group.sample(n=n, random_state=self.random_state, replace=self.replace)
                sampled_groups.append(sampled)

            sampled_df = pd.concat(sampled_groups).reset_index(drop=True)
            log.info(f"Balanced sampled {len(sampled_df)} rows across domains {self.domains}.")
            return sampled_df

        else:
            raise ValueError(f"Unknown sampling strategy: {self.strategy}")
    def save_samples(self, df: pd.DataFrame):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"{self.project_name}.json"
        df.to_json(output_path, orient="records", indent=4)
        
        un_nested_df = self.unnest_json_columns(df)
        log.info(f"{un_nested_df['image_id'].nunique()} unique images in the dataset.")
        un_nested_df.to_csv(self.output_dir / f"{self.project_name}.csv", index=False)
        log.info(f"Saved samples to {output_path}")

    def run(self):
        self.connect_db()
        try:
            self.add_category_condition()
            self.add_morphological_conditions()

            df = self.execute_query()
            df = self.parse_json_columns(df)
            df = self.unnest_json_columns(df)
            df = self.add_bbox_bin_column(df)

            sampled_df = self.sample_data(df)
            sampled_df = self.renest_json_columns(sampled_df)
            self.save_samples(sampled_df)
        finally:
            self.close_db()


@hydra.main(version_base="1.2", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    sampler = CutoutQuerySampler(cfg)
    sampler.run()


if __name__ == "__main__":
    main()
