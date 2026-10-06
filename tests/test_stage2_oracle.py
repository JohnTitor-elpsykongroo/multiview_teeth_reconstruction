import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from test_stage2_observation import make_scene
from tooth_observation.data import read_json
from tooth_observation.oracle import package_oracle
from observation_fusion.contract import load_observations, ContractError
from observation_fusion.pipeline import prepare


class OracleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.input, self.ann = make_scene(self.root / "data")
        self.output = self.root / "oracle"

    def test_annotations_are_exact_and_tissue_is_ignore(self):
        package_oracle(self.input, self.ann, self.output)
        obs = load_observations(self.output / "manifest.json", allow_oracle=True)
        self.assertEqual(obs.manifest["observation_kind"], "oracle")
        self.assertIsNone(obs.manifest["model"]["checkpoint_sha256"])
        for v in obs.views:
            self.assertTrue(np.all(v.labels[:, :2] == 255))
            self.assertTrue(np.all(v.labels[1:, 4:12] == 11))
            self.assertTrue(np.all(v.labels[1:, 12:20] == 48))
            np.testing.assert_array_equal(v.confidence, v.valid.astype(np.float32))
            self.assertEqual(v.metadata["quality"]["observed_fdi"], [11, 48])

    def test_oracle_never_silently_becomes_prediction_or_fixture(self):
        package_oracle(self.input, self.ann, self.output)
        with self.assertRaisesRegex(ContractError, "oracle"):
            load_observations(self.output / "manifest.json")
        with self.assertRaisesRegex(ContractError, "oracle"):
            load_observations(self.output / "manifest.json", allow_fixture=True)

    def test_truth_and_invisible_instance_lists_never_opened(self):
        original = Path.open
        def guard(path, *args, **kwargs):
            if "truth" in path.parts or path.name == "unused.json":
                raise AssertionError("truth/instance list accessed")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            package_oracle(self.input, self.ann, self.output)

    def test_source_preserved_and_output_not_overwritten(self):
        before = self.ann.read_bytes()
        package_oracle(self.input, self.ann, self.output)
        self.assertEqual(before, self.ann.read_bytes())
        with self.assertRaises(FileExistsError):
            package_oracle(self.input, self.ann, self.output)

    def test_stage_three_provenance_and_third_molar(self):
        package_oracle(self.input, self.ann, self.output)
        report = prepare(self.output / "manifest.json", self.root / "fusion",
                         {"kind": "assume_all_present", "origin": "test assumption"}, allow_oracle=True)
        self.assertEqual(report["observation_kind"], "oracle")
        self.assertFalse(report["debug_fixture"])
        self.assertEqual(report["excluded_third_molar_pixels"]["view_0"]["48"], 120)
        self.assertFalse(report["fit_ready"])
        self.assertEqual(read_json(self.root / "fusion/adapted/manifest.json")["observation_kind"], "oracle")


if __name__ == "__main__":
    unittest.main()
