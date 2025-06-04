import json
import sys
from collections import defaultdict, OrderedDict
from pathlib import Path

def main(species_info_path, class_groupings_path):
    # Load species_info
    with open(species_info_path, "r") as f:
        species_info = json.load(f)["species"]

    group_keys = [
        "family", "order", "genus", "subclass", "growth_habit", "group", "vegetation"
    ]

    def make_groupdict():
        return defaultdict(lambda: {"class_ids": [], "values": None})

    groupings = {key: make_groupdict() for key in group_keys}

    for sp, info in species_info.items():
        class_id = info["class_id"]
        for key in group_keys:
            groupval = str(info.get(key, "unknown")).lower()

            # Skip colorchecker as a group!
            if groupval == "colorchecker":
                continue

            # For vegetation, only two groups: background and vegetation
            if key == "vegetation":
                if groupval == "background":
                    groupings[key]["background"]["class_ids"].append(class_id)
                
                elif groupval != "colorchecker":
                    if class_id not in (0, 28):  # <-- Only add if not background/colorchecker
                        groupings[key]["vegetation"]["class_ids"].append(class_id)
            elif groupval in ("background", "unknown"):
                groupings[key][groupval]["class_ids"].append(class_id)
            else:
                groupings[key][groupval]["class_ids"].append(class_id)

    # Assign values index
    for key in group_keys:
        if key == "vegetation":
            group_order = ["background", "vegetation"]
        else:
            bg_like = [g for g in ("background", "unknown") if g in groupings[key]]
            rest = sorted([g for g in groupings[key] if g not in bg_like])
            group_order = bg_like + rest

        for i, group in enumerate(group_order):
            groupings[key][group]["values"] = i

        groupings[key] = OrderedDict((g, groupings[key][g]) for g in group_order)

    # Always include both 0 and 28 as background for every group type
    for key in group_keys:
        if "background" in groupings[key]:
            ids = set(groupings[key]["background"]["class_ids"])
            ids.update([0, 28])
            groupings[key]["background"]["class_ids"] = sorted(ids)
        # Remove colorchecker group if somehow present
        groupings[key].pop("colorchecker", None)

    # Explicitly do this for vegetation, in case upstream logic changes
    if "background" in groupings["vegetation"]:
        ids = set(groupings["vegetation"]["background"]["class_ids"])
        ids.update([0, 28])
        groupings["vegetation"]["background"]["class_ids"] = sorted(ids)


    def to_python_dict(groupings):
        out = "CLASSGROUPS = {\n"
        for key, d in groupings.items():
            out += f'    "{key}": {{\n'
            for group, vals in d.items():
                out += f'        "{group}": {{\n'
                out += f'            "class_ids": {sorted(vals["class_ids"])},\n'
                out += f'            "values": {vals["values"]}\n'
                out += f'        }},\n'
            out += f'    }},\n'
        out += "}\n"
        return out

    pytext = "# Generated CLASSGROUPS dictionary\n\n" + to_python_dict(groupings)

    # Output file: always in same dir as script or in same dir as species_info by default?
    output_path = Path(class_groupings_path)
    with open(output_path, "w") as f:
        f.write(pytext)

    print(f"New class_groupings.py generated with all species in {species_info_path}.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_class_groupings.py /path/to/species_info.json /path/to/class_groupings.py")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
