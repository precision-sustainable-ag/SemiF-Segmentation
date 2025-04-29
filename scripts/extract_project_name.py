# scripts/extract_project_name.py

import yaml
import sys

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "conf/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    project_name = cfg.get("project", {}).get("name", "unknown_project")
    print(project_name)

if __name__ == "__main__":
    main()
