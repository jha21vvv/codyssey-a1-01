"""One-click evaluator verification runner for codyssey-a1-01."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import DataAnalyzer  # noqa: E402


FORBIDDEN_LIBS = (
    "cv2",
    "PIL",
    "sklearn",
    "nltk",
    "pandas_profiling",
    "ydata_profiling",
    "sweetviz",
)


def check_forbidden_imports() -> list[str]:
    violations: list[str] = []
    targets = [ROOT / "src" / "pipeline.py", ROOT / "notebooks" / "analysis_report.ipynb"]
    for path in targets:
        if not path.exists():
            violations.append(f"Missing file: {path}")
            continue
        text = path.read_text(encoding="utf-8")
        for lib in FORBIDDEN_LIBS:
            if re.search(rf"\b(import|from)\s+{re.escape(lib)}\b", text):
                violations.append(f"Forbidden import '{lib}' in {path.name}")
    return violations


def check_markdown_relative_images() -> list[str]:
    issues: list[str] = []
    for md_name in ("README.md", "스터디.md"):
        md_path = ROOT / md_name
        if not md_path.exists():
            continue
        text = md_path.read_text(encoding="utf-8")
        if "file:///" in text:
            issues.append(f"Absolute file:/// link found in {md_name}")
    return issues


def main() -> int:
    t0 = time.perf_counter()
    analyzer = DataAnalyzer(
        csv_path=ROOT / "data" / "ecommerce_transactions.csv",
        images_path=ROOT / "data" / "product_images.npy",
    )
    df, images = analyzer.load_data()
    analyzer.extract_multimodal_features(downsample_step=2)
    analyzer.handle_missing_values(group_col="category", target_cols=["unit_price"], strategy="median")
    price_out = analyzer.detect_outliers(column="unit_price", iqr_multiplier=1.5, action="cap")
    qty_out = analyzer.detect_outliers(column="quantity", iqr_multiplier=1.5, action="cap")
    stats = analyzer.compute_statistics(["unit_price", "quantity", "amount", "img_mean", "img_std", "word_count", "name_length"])
    rfm = analyzer.calculate_rfm()

    violations = check_forbidden_imports()
    md_issues = check_markdown_relative_images()
    elapsed = round(time.perf_counter() - t0, 4)

    seg_counts = rfm["Segment"].value_counts().to_dict()
    checks = {
        "rows_ge_1000": len(df) >= 1000,
        "cols_ge_8": df.shape[1] >= 8,
        "images_4d_numpy": images is not None and images.ndim == 4,
        "missing_imputed": int(analyzer.df["unit_price"].isna().sum()) == 0,
        "outliers_handled": price_out["outlier_count"] > 0 and qty_out["outlier_count"] > 0,
        "rfm_segments_ge_4": len(seg_counts) >= 4,
        "no_forbidden_imports": len(violations) == 0,
        "relative_md_images": len(md_issues) == 0,
    }

    print("=" * 64)
    print("codyssey-a1-01 Evaluator Requirement Verification")
    print("=" * 64)
    for key, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {key}")

    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "elapsed_sec": elapsed,
        "dataset_rows": int(len(df)),
        "dataset_cols": int(df.shape[1]),
        "image_shape": list(images.shape) if images is not None else None,
        "outliers_capped": {
            "unit_price": price_out["outlier_count"],
            "quantity": qty_out["outlier_count"],
        },
        "corr_img_mean_std": round(float(stats["corr_matrix"].loc["img_mean", "img_std"]), 4),
        "corr_word_name_len": round(float(stats["corr_matrix"].loc["word_count", "name_length"]), 4),
        "rfm_segments": {k: int(v) for k, v in seg_counts.items()},
        "checks": checks,
    }
    print("-" * 64)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("=" * 64)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
