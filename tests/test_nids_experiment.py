import unittest

import numpy as np

from src.nids_experiment import binary_metrics, clean_columns, normalize_label


class PipelineTests(unittest.TestCase):
    def test_normalize_label_repairs_web_attack_encoding(self):
        self.assertEqual(
            normalize_label(" Web Attack \ufffd Brute Force "),
            "Web Attack - Brute Force",
        )

    def test_normalize_label_standardizes_benign(self):
        self.assertEqual(normalize_label(" BENIGN "), "Benign")

    def test_clean_columns_collapses_header_whitespace(self):
        self.assertEqual(
            clean_columns([" Flow  Duration ", " Label"]),
            ["Flow Duration", "Label"],
        )

    def test_binary_metrics_uses_attack_as_positive_class(self):
        metrics = binary_metrics(
            np.array([0, 0, 1, 1]),
            np.array([0, 1, 1, 0]),
            np.array([0.1, 0.6, 0.8, 0.4]),
        )
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertAlmostEqual(metrics["false_positive_rate"], 0.5)
        self.assertAlmostEqual(metrics["false_negative_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()

