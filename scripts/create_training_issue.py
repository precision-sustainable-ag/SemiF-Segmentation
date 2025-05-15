import os
import pandas as pd
import requests
import json
from pathlib import Path
from datetime import datetime


def format_run_info(run_info_path: str) -> str:
    if not os.path.exists(run_info_path):
        return "_run_info.json not found_"

    with open(run_info_path) as f:
        info = json.load(f)

    lines = ["| Key | Value |", "|------|-------|"]
    for k, v in info.items():
        if k == "version_dir":
            version_dir = Path(v)
            v = version_dir.relative_to(os.getcwd())
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines), info

def extract_metrics(csv_path):
    if not os.path.exists(csv_path):
        return "N/A", "N/A", "_No metrics file found._"

    df = pd.read_csv(csv_path)

    # Create training and validation subsets
    train_df = df[df["train_dataset_iou"].notnull()].copy()
    val_df = df[df["valid_dataset_iou"].notnull()].copy()

    # Merge on epoch and step
    merged = pd.merge(
        train_df[[
            "epoch", 
            "step",
            "train_loss",
            "train_class_0_iou",
            "train_class_1_iou",
            "train_class_2_iou",
            "train_dataset_iou",
            "train_per_image_iou"
        ]],
        val_df[[
            "epoch", "step", 
            "valid_loss",
            "valid_class_0_iou",
            "valid_class_1_iou",
            "valid_class_2_iou",
            "valid_dataset_iou",
            "valid_per_image_iou"
            
        ]],
        on=["epoch", "step"],
        how="outer"
    ).sort_values("epoch")

    # Round float columns and fill NaNs
    for col in merged.columns:
        if merged[col].dtype == float:
            merged[col] = merged[col].round(4)
    merged.fillna("-", inplace=True)

    # Find latest validation row for summary values
    last_valid = val_df.dropna(subset=["valid_dataset_iou", "valid_per_image_iou"]).tail(1)
    dataset_iou = round(float(last_valid["valid_dataset_iou"].values[0]), 4) if not last_valid.empty else "N/A"
    per_image_iou = round(float(last_valid["valid_per_image_iou"].values[0]), 4) if not last_valid.empty else "N/A"

    # Select only last 10 rows for display
    markdown_table = merged.tail(10).to_markdown(index=False)

    return dataset_iou, per_image_iou, markdown_table


def create_issue(repo, token, title, body):
    url = f"https://api.github.com/repos/{repo}/issues"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json"
    }
    payload = {
        "title": title,
        "body": body,
        "labels": ["training"]
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        print("✅ Issue created successfully!")
    else:
        print("❌ Failed to create issue:", response.status_code)
        print(response.text)
        exit(1)

def main():
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    token = os.environ.get("GH_TOKEN")
    version_dir = os.environ.get("VERSION_DIR", "")
    project_name = os.environ.get("PROJECT_NAME", "unknown_project")
    
    image_url = f"https://raw.githubusercontent.com/{repo}/training-artifacts/assets/metrics_{run_id}.png"
    prediction_url = f"https://raw.githubusercontent.com/{repo}/training-artifacts/assets/predictions_{run_id}.png"

    inf0 = f"https://raw.githubusercontent.com/{repo}/training-artifacts/assets/inference0_{run_id}.png"
    inf1 = f"https://raw.githubusercontent.com/{repo}/training-artifacts/assets/inference1_{run_id}.png"
    inf2 = f"https://raw.githubusercontent.com/{repo}/training-artifacts/assets/inference2_{run_id}.png"


    image_md = f"![Training Plots]({image_url})"
    prediction_md = f"![Sample Prediction]({prediction_url})"

    inf0_md = f"![Inference 0]({inf0})"
    inf1_md = f"![Inference 1]({inf1})"
    inf2_md = f"![Inference 2]({inf2})"

    run_info_file = "logs/run_info.json"
    run_info_md, info = format_run_info(run_info_file)
    
    metrics_file = os.path.join(version_dir, "metrics.csv")

    _, _, markdown_table = extract_metrics(metrics_file)
    
    version = Path(info["version_dir"]).stem
    model_name = info["model_name"]
    encoder = info["encoder"]
    

    title = f"Training Report - {project_name} - {version} - {model_name} - {encoder}"
    body = f"""
## Segmentation Model Training Report

**Project:** `{project_name}`  
**Run Time:** `{datetime.now().isoformat()}`


### Run Info Summary

{run_info_md}

---

#### Last 10 Training Steps

{markdown_table}

---

#### Loss and IoU

{image_md}

#### Test Set Prediction

{prediction_md}

#### Real-World Predictions

{inf0_md}
{inf1_md}
{inf2_md}

---
_Triggered by push to `develop` branch._
"""

    create_issue(repo, token, title, body)


if __name__ == "__main__":
    main()