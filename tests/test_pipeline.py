"""Fast unit tests for src.pipeline.DataAnalyzer (Step 3)."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.pipeline import DataAnalyzer


class TestDataAnalyzer(unittest.TestCase):
    """Unit tests covering normal flows and edge cases of DataAnalyzer."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp_dir.name)
        self.csv_path = self.base / "sample.csv"
        self.npy_path = self.base / "images.npy"

        df = pd.DataFrame(
            {
                "product_id": [1, 2, 3, 4, 5, 6, 7, 8],
                "product_name": [
                    "Blue Cotton Shirt",
                    "Black Leather Bag",
                    "White Running Shoes",
                    "Red Summer Dress",
                    "Denim Casual Jeans",
                    "Silver Classic Watch",
                    "Green Sport Cap",
                    "Beige Wool Scarf",
                ],
                "category": [
                    "Apparel",
                    "Accessories",
                    "Footwear",
                    "Apparel",
                    "Apparel",
                    "Accessories",
                    "Accessories",
                    "Apparel",
                ],
                "customer_id": ["C1", "C1", "C2", "C2", "C3", "C4", "C4", "C1"],
                "order_date": [
                    "2025-01-10",
                    "2025-06-15",
                    "2025-02-20",
                    "2025-05-01",
                    "2025-03-05",
                    "2025-11-25",
                    "2025-12-01",
                    "2025-12-10",
                ],
                "unit_price": [20.0, 40.0, 60.0, np.nan, 25.0, 500.0, 35.0, 22.0],
                "quantity": [1, 2, 1, 2, 1, 1, 3, 2],
                "amount": [20.0, 80.0, 60.0, np.nan, 25.0, 500.0, 105.0, 44.0],
            }
        )
        df.to_csv(self.csv_path, index=False)

        rng = np.random.default_rng(0)
        images = rng.integers(0, 255, size=(8, 16, 12, 3), dtype=np.uint8)
        np.save(self.npy_path, images)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_end_to_end_pipeline(self) -> None:
        analyzer = DataAnalyzer(self.csv_path, self.npy_path)
        df, imgs = analyzer.load_data()
        self.assertEqual(df.shape, (8, 8))
        self.assertEqual(imgs.shape, (8, 16, 12, 3))

        df_feat = analyzer.extract_multimodal_features(downsample_step=2)
        self.assertIn("img_mean", df_feat.columns)
        self.assertIn("img_std", df_feat.columns)
        self.assertIn("img_downsampled_mean", df_feat.columns)
        self.assertIn("word_count", df_feat.columns)
        self.assertIn("name_length", df_feat.columns)
        self.assertEqual(df_feat.loc[0, "word_count"], 3)

        df_imp = analyzer.handle_missing_values(group_col="category", target_cols=["unit_price"])
        self.assertEqual(df_imp["unit_price"].isna().sum(), 0)
        self.assertEqual(df_imp["amount"].isna().sum(), 0)
        # Apparel median of [20.0, 25.0, 22.0] is 22.0
        self.assertAlmostEqual(df_imp.loc[3, "unit_price"], 22.0)

        report = analyzer.detect_outliers(column="unit_price", iqr_multiplier=1.5, action="cap")
        self.assertGreaterEqual(report["outlier_count"], 1)
        self.assertLessEqual(analyzer.df["unit_price"].max(), report["upper_bound"])

        stats = analyzer.compute_statistics(["unit_price", "quantity", "amount"])
        self.assertIn("summary", stats)
        self.assertIn("corr_matrix", stats)

        rfm = analyzer.calculate_rfm()
        self.assertEqual(len(rfm), 4)
        self.assertTrue(set(rfm["Segment"]).issubset({"VIP", "Loyal", "New", "Churned"}))

    def test_methods_before_load_raise(self) -> None:
        analyzer = DataAnalyzer(self.csv_path)
        with self.assertRaises(ValueError):
            analyzer.extract_multimodal_features()
        with self.assertRaises(ValueError):
            analyzer.handle_missing_values()
        with self.assertRaises(ValueError):
            analyzer.detect_outliers()
        with self.assertRaises(ValueError):
            analyzer.calculate_rfm()

    def test_missing_files_raise(self) -> None:
        with self.assertRaises(FileNotFoundError):
            DataAnalyzer(self.base / "nope.csv").load_data()
        with self.assertRaises(FileNotFoundError):
            DataAnalyzer(self.csv_path, self.base / "nope.npy").load_data()

    def test_parse_image_array_fromstring_and_literal(self) -> None:
        arr1 = DataAnalyzer.parse_image_array("10 20 30 40", shape=(2, 2))
        self.assertEqual(arr1.shape, (2, 2))
        self.assertEqual(arr1[1, 1], 40)

        arr2 = DataAnalyzer.parse_image_array("[[1, 2], [3, 4]]")
        self.assertEqual(arr2.shape, (2, 2))
        self.assertEqual(arr2[0, 1], 2)

    def test_iqr_remove_syncs_images(self) -> None:
        analyzer = DataAnalyzer(self.csv_path, self.npy_path)
        analyzer.load_data()
        analyzer.handle_missing_values()
        report = analyzer.detect_outliers(column="unit_price", action="remove")
        self.assertEqual(len(analyzer.df), 8 - report["outlier_count"])
        self.assertEqual(len(analyzer.images), len(analyzer.df))

    def test_invalid_parameters_raise(self) -> None:
        analyzer = DataAnalyzer(self.csv_path)
        analyzer.load_data()
        with self.assertRaises(ValueError):
            analyzer.extract_multimodal_features(downsample_step=0)
        with self.assertRaises(ValueError):
            analyzer.handle_missing_values(strategy="mode")
        with self.assertRaises(ValueError):
            analyzer.detect_outliers(action="drop")
        with self.assertRaises(KeyError):
            analyzer.detect_outliers(column="nonexistent")


if __name__ == "__main__":
    unittest.main()
