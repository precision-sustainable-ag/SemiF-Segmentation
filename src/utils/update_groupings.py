import json
from collections import defaultdict
from pathlib import Path
import hydra
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

class ClassGroupsGenerator:
    """
    Class to generate CLASSGROUPS dictionary from species information
    and save it as a Python file.
    """

    def __init__(self, include_colorchecker: bool):
        """
        Initialize the generator with configuration options.

        :param include_colorchecker: Whether to include `colorchecker` in the background.
        """
        self.include_colorchecker = include_colorchecker
        self.classgroups = defaultdict(lambda: {"background": {"class_ids": [0], "values": 0}})

    def load_species_data(self, input_file: Path):
        """
        Load species data from a JSON file.

        :param input_file: Path to the input JSON file.
        """
        log.info(f"Loading species data from {input_file}")
        with open(input_file, "r") as file:
            self.species_data = json.load(file)

    def generate_classgroups(self):
        """
        Generate the CLASSGROUPS dictionary from the species data.
        """
        log.info("Generating CLASSGROUPS dictionary.")
        vegetation_class_ids = []

        for key, value in self.species_data["species"].items():
            class_id = value["class_id"]

            # Conditionally add colorchecker to background
            if class_id == 28 and self.include_colorchecker:
                for group_name in self.classgroups:
                    if class_id not in self.classgroups[group_name]["background"]["class_ids"]:
                        self.classgroups[group_name]["background"]["class_ids"].append(class_id)
                continue  # Skip further processing for colorchecker

            if key.lower() == "background":
                continue  # Skip further processing for background

            # Collect vegetation-related classes
            if value["group"].lower() in ["monocot", "dicot"]:
                vegetation_class_ids.append(class_id)

            # Update hierarchical groups in CLASSGROUPS
            for group_name in ["family", "order", "genus", "subclass", "growth_habit", "group"]:
                group_value = value[group_name].lower()  # Normalize to lowercase

                # Ensure the group exists in CLASSGROUPS
                if group_value not in self.classgroups[group_name]:
                    # Determine the next `values` index (avoiding skipping)
                    next_value = max(
                        entry["values"] for entry in self.classgroups[group_name].values()
                    ) + 1 if self.classgroups[group_name] else 1
                    self.classgroups[group_name][group_value] = {"class_ids": [], "values": next_value}

                # Append the class_id to the appropriate group
                if class_id not in self.classgroups[group_name][group_value]["class_ids"]:
                    self.classgroups[group_name][group_value]["class_ids"].append(class_id)

        # Ensure colorchecker is always in the background if enabled
        if self.include_colorchecker:
            for group_name in self.classgroups:
                if 28 not in self.classgroups[group_name]["background"]["class_ids"]:
                    self.classgroups[group_name]["background"]["class_ids"].append(28)

        # Add custom vegetation grouping
        self.add_custom_vegetation_grouping(vegetation_class_ids)

        # Rearrange the values for the "group" group_name
        if "group" in self.classgroups:
            if "monocot" in self.classgroups["group"]:
                self.classgroups["group"]["monocot"]["values"] = 1
            if "dicot" in self.classgroups["group"]:
                self.classgroups["group"]["dicot"]["values"] = 2

        # Convert defaultdict back to a normal dictionary
        self.classgroups = {key: dict(value) for key, value in self.classgroups.items()}

    def add_custom_vegetation_grouping(self, vegetation_class_ids):
        """
        Add a custom grouping with only background and vegetation to CLASSGROUPS.

        :param vegetation_class_ids: List of class IDs identified as vegetation.
        """
        log.info("Adding custom vegetation grouping to CLASSGROUPS.")
        self.classgroups["vegetation"] = {
            "background": {"class_ids": [0], "values": 0},
            "vegetation": {"class_ids": vegetation_class_ids, "values": 1},
        }

    def save_classgroups(self, output_file: Path):
        """
        Save the CLASSGROUPS dictionary to a Python file.

        :param output_file: Path to the output Python file.
        """
        log.info(f"Saving CLASSGROUPS dictionary to {output_file}")
        with open(output_file, "w") as file:
            file.write("# Generated CLASSGROUPS dictionary\n\n")
            file.write("CLASSGROUPS = ")
            file.write(json.dumps(self.classgroups, indent=4))
            file.write("\n")

    def print_classgroups_summary(self):
        """
        Print a summary of the generated CLASSGROUPS dictionary.
        """
        for group_name, group_data in self.classgroups.items():
            print(f"{group_name}:")
            for subgroup_name, subgroup_data in group_data.items():
                print(f"  {subgroup_name}: {subgroup_data['class_ids']}")


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    log.info("Updating class grouping options.")
    generator = ClassGroupsGenerator(include_colorchecker=cfg.task.update_groupings.colorchecker_to_background)
    generator.load_species_data(Path(cfg.paths.species_info))
    generator.generate_classgroups()
    generator.save_classgroups(Path(cfg.paths.python_grouping_file))
    generator.print_classgroups_summary()
    log.info("CLASSGROUPS generation complete.")


if __name__ == "__main__":
    main()
