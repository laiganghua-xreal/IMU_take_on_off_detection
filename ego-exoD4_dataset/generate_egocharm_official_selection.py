#!/usr/bin/env python3
"""Generate a size-bounded EgoCHARM subset using official Ego-Exo4D splits."""

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = [
    "Basketball",
    "Soccer",
    "Dance",
    "Rock Climbing",
    "Cooking",
    "Bike Repair",
    "Music",
]
SPLITS = ["train", "val", "test"]
DEFAULT_TARGET_GB = 280.0
SEED = 42


def build_selection(takes, manifest, target_bytes):
    manifest_by_uid = {item["uid"]: item for item in manifest}
    sizes = {
        uid: sum(path.get("size", 0) for path in item.get("paths", []))
        for uid, item in manifest_by_uid.items()
    }

    grouped = defaultdict(lambda: defaultdict(list))
    for take in takes:
        uid = take.get("take_uid")
        activity = take.get("parent_task_name")
        participant = take.get("participant_uid")
        official_splits = manifest_by_uid.get(uid, {}).get("splits", [])

        eligible = (
            activity in CLASSES
            and participant is not None
            and take.get("has_trimmed_vrs")
            and take.get("validated")
            and not take.get("is_dropped")
            and float(take.get("duration_sec") or 0) >= 30
            and uid in sizes
            and len(official_splits) == 1
            and official_splits[0] in SPLITS
        )
        if eligible:
            grouped[(activity, official_splits[0])][str(participant)].append(take)

    stratum_sizes = {
        key: sum(
            sizes[take["take_uid"]]
            for participant_takes in participants.values()
            for take in participant_takes
        )
        for key, participants in grouped.items()
    }
    total_available = sum(stratum_sizes.values())
    if total_available == 0:
        raise ValueError("No eligible takes were found in the metadata and manifest")

    selected = []
    for activity in CLASSES:
        for split in SPLITS:
            key = (activity, split)
            if key not in stratum_sizes:
                continue

            budget = target_bytes * stratum_sizes[key] / total_available
            participant_groups = []
            for participant, participant_takes in grouped[key].items():
                group_size = sum(sizes[take["take_uid"]] for take in participant_takes)
                stable_order = hashlib.sha256(
                    f"EgoCHARM-official:{SEED}:{activity}:{split}:{participant}".encode()
                ).hexdigest()
                participant_groups.append(
                    (stable_order, group_size, participant_takes)
                )

            used = 0
            for _, group_size, participant_takes in sorted(participant_groups):
                if used + group_size <= budget:
                    used += group_size
                    selected.extend(participant_takes)

    entries = []
    for take in selected:
        uid = take["take_uid"]
        split = manifest_by_uid[uid]["splits"][0]
        relative_path = str(
            Path(take.get("root_dir", ""))
            / take.get("vrs_relative_path", "aria01_noimagestreams.vrs")
        )
        entries.append(
            {
                "take_uid": uid,
                "take_name": take["take_name"],
                "participant_uid": take["participant_uid"],
                "class": take["parent_task_name"],
                "official_split": split,
                "duration_sec": take["duration_sec"],
                "size_bytes": sizes[uid],
                "relative_path": relative_path,
            }
        )

    entries.sort(
        key=lambda item: (
            SPLITS.index(item["official_split"]),
            CLASSES.index(item["class"]),
            str(item["participant_uid"]),
            item["take_uid"],
        )
    )

    split_summary = {}
    for split in SPLITS:
        split_entries = [entry for entry in entries if entry["official_split"] == split]
        split_summary[split] = {
            "takes": len(split_entries),
            "participants": len(
                {entry["participant_uid"] for entry in split_entries}
            ),
            "duration_sec": sum(entry["duration_sec"] for entry in split_entries),
            "size_bytes": sum(entry["size_bytes"] for entry in split_entries),
            "classes": dict(Counter(entry["class"] for entry in split_entries)),
        }

    return {
        "dataset": "Ego-Exo4D",
        "release": "v2",
        "split_policy": "official",
        "sampling_seed": SEED,
        "target_bytes": int(target_bytes),
        "actual_bytes": sum(entry["size_bytes"] for entry in entries),
        "num_takes": len(entries),
        "num_participants": len(
            {entry["participant_uid"] for entry in entries}
        ),
        "split_summary": split_summary,
        "entries": entries,
    }


def parse_args():
    dataset_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=dataset_root / "metadata" / "takes.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=dataset_root / "metadata" / "take_vrs_noimagestream_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=dataset_root / "metadata" / "egocharm_official_selection.json",
    )
    parser.add_argument("--target-gb", type=float, default=DEFAULT_TARGET_GB)
    parser.add_argument(
        "--print-uids",
        action="store_true",
        help="Print selected take UIDs to stdout for use with egoexo --uids",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.metadata.open(encoding="utf-8") as file:
        takes = json.load(file)
    with args.manifest.open(encoding="utf-8") as file:
        manifest = json.load(file)

    selection = build_selection(
        takes,
        manifest,
        target_bytes=int(args.target_gb * 1_000_000_000),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(selection, file, ensure_ascii=False, indent=2)

    print(
        f"Wrote {selection['num_takes']} takes "
        f"({selection['actual_bytes'] / 1_000_000_000:.2f} GB) "
        f"to {args.output}",
        file=sys.stderr,
    )
    if args.print_uids:
        for entry in selection["entries"]:
            print(entry["take_uid"])


if __name__ == "__main__":
    main()
