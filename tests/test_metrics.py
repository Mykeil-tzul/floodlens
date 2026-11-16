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
