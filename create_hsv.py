"""Create a dedicated court1 HSV trial config without modifying existing files.

Usage from the tennisgametracker project directory:
    python court1_hsv_preset.py

Reads hsv_config_new.json and writes hsv_config_court1_trial.json.
It preserves all existing court geometry, serve settings, and unrelated fields.
"""

import json
from pathlib import Path

SOURCE = Path("hsv_config_new.json")
DESTINATION = Path("hsv_config_court1_trial.json")

# Experimental ranges derived from manually clicked court1 ball pixels,
# frames 35042-35067 (and their 3x3 neighborhoods), NOT video-validated.
HSV_FILTERS = {
    "regular_court": (40, 110, 10, 110, 30, 250),
    "alt1": (90, 110, 25, 95, 65, 190),
    "alt2": (40, 65, 8, 105, 95, 255),
    "alt3": (95, 110, 20, 95, 120, 255),
    "alt4": (43, 65, 15, 110, 30, 125),
    "alt5": (85, 110, 20, 95, 40, 130),
    "alt6": (40, 65, 10, 110, 30, 250),
    "s_30": (40, 65, 10, 110, 130, 255),
    "h_10": (40, 110, 10, 110, 30, 125),
    "alts_20": (43, 70, 5, 55, 145, 255),
    "alts9_11": (90, 110, 20, 95, 75, 255),
    "behind_net": (40, 110, 10, 110, 30, 250),
}

KEYS = ("h_min", "h_max", "s_min", "s_max", "v_min", "v_max")


def main():
    if not SOURCE.is_file():
        raise SystemExit(f"Cannot find {SOURCE.resolve()}. Run from your project folder.")
    if DESTINATION.exists():
        raise SystemExit(
            f"Refusing to overwrite {DESTINATION.resolve()}. "
            "Rename or remove that trial file first if you want to regenerate it."
        )

    config = json.loads(SOURCE.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise SystemExit("Source config JSON must contain an object.")

    for name, values in HSV_FILTERS.items():
        previous = config.get(name, {})
        if previous is not None and not isinstance(previous, dict):
            raise SystemExit(f"Expected filter {name!r} to be a JSON object.")
        updated = dict(previous or {})
        updated.update(zip(KEYS, values))
        config[name] = updated

    DESTINATION.write_text(json.dumps(config, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created {DESTINATION.resolve()} using {SOURCE}.")
    print(f"Changed {len(HSV_FILTERS)} HSV ranges; preserved court/serve geometry and other settings.")
    print("Trial preset only. The camera image and non-HSV serve-stance validation still need checking.")


if __name__ == "__main__":
    main()