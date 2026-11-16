import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
from train import report, scores


class MetricsTests(unittest.TestCase):
    def test_pixel_metrics_ignore_nodata(self):
        prediction = np.array([[True, True], [False, True]])
        target = np.array([[1, 0], [1, -1]])
        valid = target >= 0
        self.assertEqual(report(scores(prediction, target, valid)), {
            "iou": 1/3, "precision": .5, "recall": .5, "tp": 1, "fp": 1, "fn": 1,
        })

    def test_final_report_uses_the_frozen_checkpoint_and_only_sweden_tiles(self):
        root = Path(__file__).resolve().parents[1]
        plan = json.loads((root / "final-test-plan.json").read_text())
        result = json.loads((root / "site/final-results.json").read_text())
        self.assertEqual(hashlib.sha256((root / "model.pt").read_bytes()).hexdigest(),
                         plan["checkpoint_sha256"])
        for field in ("event", "checkpoint_sha256", "model_cutoff", "threshold_cutoff"):
            self.assertEqual(result[field], plan[field])
        self.assertEqual(len(result["tiles"]), 30)
        self.assertTrue(all(tile["id"].startswith("EMSR427-") for tile in result["tiles"]))
        for method in ("model", "threshold"):
            for count in ("tp", "fp", "fn"):
                self.assertEqual(result["metrics"][method][count],
                                 sum(tile[method][count] for tile in result["tiles"]))
