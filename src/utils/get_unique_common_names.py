import sqlite3
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig

log = logging.getLogger(__name__)

@hydra.main(version_base="1.2", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    db_path = cfg.paths.db.db_file
    table_name = "semif_cutouts"

    # Connect to SQLite DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    log.info(f"Connected to database: {db_path}")

    try:
        # Query to extract unique common_name values from JSON column
        query = f"""
            SELECT DISTINCT LOWER(TRIM(json_extract(category, '$.common_name')))
            FROM {table_name}
            WHERE json_extract(category, '$.common_name') IS NOT NULL
        """
        cursor.execute(query)
        results = cursor.fetchall()
        common_names = sorted(set(name[0] for name in results if name[0]))

        log.info(f"Found {len(common_names)} unique common names.")
        for name in common_names:
            print(name)

    except sqlite3.Error as e:
        log.error(f"Error querying database: {e}")
    finally:
        conn.close()
        log.info("Database connection closed.")

if __name__ == "__main__":
    main()
