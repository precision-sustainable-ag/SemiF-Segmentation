import json
import logging
import uuid
from pathlib import Path

import pandas as pd
import hydra
from omegaconf import DictConfig

from src.utils.sql3_query import SQLiteQueryHandler

log = logging.getLogger(__name__)

def flatten_dict(d, parent_key='', sep='.'):
    """Recursively flattens a nested dictionary using dot notation."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def convert_to_csv(documents, csv_path):
    """Converts a list of nested dictionaries to a flat CSV using pandas."""
    flattened = [flatten_dict(d) for d in documents]
    df = pd.DataFrame(flattened)
    df.to_csv(csv_path, index=False)
    return df  # Optional: return for inspection or testing


def recursively_parse_json(obj):
    """
    Recursively convert JSON strings in a nested structure (dict or list) into Python objects.
    
    Args:
        obj: The object to process (can be dict, list, or a primitive value).
    
    Returns:
        The object with any JSON strings parsed into dictionaries/lists.
    """
    if isinstance(obj, str):
        # Try to parse the string as JSON. If it fails, return the original string.
        try:
            parsed = json.loads(obj)
            # Recursively process the parsed object.
            return recursively_parse_json(parsed)
        except json.JSONDecodeError:
            return obj  # Not a JSON string, return as-is.
    elif isinstance(obj, dict):
        return {key: recursively_parse_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [recursively_parse_json(item) for item in obj]
    else:
        # For any other data type, return it unchanged.
        return obj
    
@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to initialize the MongoDBRecipeManager and start the recipe creation process.
    """
    log.info("Starting recipe creation process.")  # Log the start of the process

    query_handler = SQLiteQueryHandler(cfg)
    query_handler.add_conditions()
    rows, columns = query_handler.execute_query()
    query_handler.close()
    log.info(f"Retrieved {len(rows)} documents from the database.")
    # Convert the rows to a list of dictionaries.
    documents = [dict(zip(columns, row)) for row in rows]
    # Ensure each document has an _id field.
    for doc in documents:
        if "_id" not in doc:
            # Generate a new unique identifier as a string.
            doc["_id"] = str(uuid.uuid4())
    # Convert nested JSON strings into dictionaries/lists.
    documents = [recursively_parse_json(doc) for doc in documents]
    save_json = {"cutouts": documents}
    output_path = Path(cfg.paths.project_mode_dir) / f"{cfg.project.name}.csv"
    # Save the list of documents to a json file.
    with open(output_path.with_suffix('.json'), 'w') as json_file:
        json.dump(save_json, json_file, indent=4)
    convert_to_csv(documents, output_path)
    log.info(f"Converted documents to CSV and saved to {output_path}.")
    
    log.info("Recipe creation completed.")

if __name__ == "__main__":
    main()
