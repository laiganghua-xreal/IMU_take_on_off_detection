import importlib.util
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "generate_egocharm_official_selection.py"
)

spec = importlib.util.spec_from_file_location("selection_generator", SCRIPT_PATH)
selection_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection_generator)


class GenerateSelectionTests(unittest.TestCase):
    def test_filters_invalid_takes_and_preserves_official_split(self):
        takes = [
            self.take("take-a", "Basketball", 1),
            self.take("take-b", "Health", 2),
            self.take("take-c", "Soccer", 3, validated=False),
            self.take("take-d", "Dance", 4, duration_sec=20),
        ]
        manifest = [
            self.manifest_item("take-a", 100, "train"),
            self.manifest_item("take-b", 100, "val"),
            self.manifest_item("take-c", 100, "test"),
            self.manifest_item("take-d", 100, "train"),
        ]

        selection = selection_generator.build_selection(
            takes, manifest, target_bytes=1_000
        )

        self.assertEqual([entry["take_uid"] for entry in selection["entries"]], ["take-a"])
        self.assertEqual(selection["entries"][0]["official_split"], "train")
        self.assertEqual(selection["actual_bytes"], 100)

    def test_budgeting_keeps_each_participant_group_whole(self):
        takes = [
            self.take("take-a1", "Basketball", 10),
            self.take("take-a2", "Basketball", 10),
            self.take("take-b1", "Basketball", 20),
            self.take("take-c1", "Cooking", 30),
            self.take("take-d1", "Cooking", 40),
        ]
        manifest = [
            self.manifest_item("take-a1", 100, "train"),
            self.manifest_item("take-a2", 100, "train"),
            self.manifest_item("take-b1", 100, "train"),
            self.manifest_item("take-c1", 100, "test"),
            self.manifest_item("take-d1", 100, "test"),
        ]

        selection = selection_generator.build_selection(
            takes, manifest, target_bytes=300
        )

        selected_by_participant = {}
        for entry in selection["entries"]:
            selected_by_participant.setdefault(entry["participant_uid"], set()).add(
                entry["take_uid"]
            )

        self.assertLessEqual(selection["actual_bytes"], 300)
        self.assertNotEqual(selected_by_participant.get(10), {"take-a1"})
        self.assertNotEqual(selected_by_participant.get(10), {"take-a2"})

    @staticmethod
    def take(uid, activity, participant, validated=True, duration_sec=60):
        return {
            "take_uid": uid,
            "take_name": uid,
            "participant_uid": participant,
            "parent_task_name": activity,
            "has_trimmed_vrs": True,
            "validated": validated,
            "is_dropped": False,
            "duration_sec": duration_sec,
            "vrs_relative_path": "aria01_noimagestreams.vrs",
        }

    @staticmethod
    def manifest_item(uid, size, split):
        return {
            "uid": uid,
            "paths": [{"size": size}],
            "splits": [split],
        }


if __name__ == "__main__":
    unittest.main()
