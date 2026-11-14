"""Visualization module for EDA 6-chart suite and IQR outlier comparison."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.pipeline import DataAnalyzer


def generate_eda_charts(
    csv_path: str | Path = ROOT / "data" / "ecommerce_transactions.csv",
    images_path: str | Path = ROOT / "data" / "product_images.npy",
    out_dir: str | Path = ROOT / "diagrams",
) -> tuple[Path, Path]:
    """Run DataAnalyzer pipeline and save outlier boxplot comparison + 6-chart EDA PNGs."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid")

    analyzer = DataAnalyzer(csv_path=csv_path, images_path=images_path)
    analyzer.load_data()
    analyzer.extract_multimodal_features(downsample_step=2)
    df_imputed = analyzer.handle_missing_values(
        group_col="category", target_cols=["unit_price"], strategy="median"
    )

    price_before = df_imputed["unit_price"].copy()
    analyzer.detect_outliers(column="unit_price", iqr_multiplier=1.5, action="cap")
    analyzer.detect_outliers(column="quantity", iqr_multiplier=1.5, action="cap")
    price_after = analyzer.df["unit_price"].copy()

    df_clean = analyzer.df.copy()
    num_cols = ["unit_price", "quantity", "amount", "img_mean", "img_std", "word_count", "name_length"]
    stats = analyzer.compute_statistics(numeric_cols=num_cols)
    rfm = analyzer.calculate_rfm()

    # 1) Dedicated same-scale before vs after IQR boxplot
    bp_df = pd.DataFrame(
        {"Before IQR": price_before, "After IQR": price_after}
    ).melt(var_name="Stage", value_name="Unit Price (USD)")

    fig_bp, ax_bp = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=bp_df, x="Stage", y="Unit Price (USD)", ax=ax_bp, color="#60A5FA")
    ax_bp.set_title("Unit Price Distribution: Before vs After IQR Capping (Same Scale)")
    ax_bp.set_xlabel("Preprocessing Stage")
    ax_bp.set_ylabel("Unit Price (USD)")
    boxplot_file = out_path / "outlier_boxplot_comparison.png"
    fig_bp.savefig(boxplot_file, dpi=150, bbox_inches="tight")
    plt.close(fig_bp)

    # 2) Full 6-chart EDA grid
    fig, axes = plt.subplots(3, 2, figsize=(15, 16))

    # Chart 1: Histogram
    sns.histplot(df_clean["amount"], bins=30, kde=True, ax=axes[0, 0], color="#2563EB")
    axes[0, 0].set_title("1. Histogram: Order Amount Distribution")
    axes[0, 0].set_xlabel("Order Amount (USD)")
    axes[0, 0].set_ylabel("Frequency")

    # Chart 2: Boxplot (Before vs After IQR)
    sns.boxplot(data=bp_df, x="Stage", y="Unit Price (USD)", ax=axes[0, 1], color="#60A5FA")
    axes[0, 1].set_title("2. Boxplot: Unit Price Before vs After IQR Capping")
    axes[0, 1].set_xlabel("Preprocessing Stage")
    axes[0, 1].set_ylabel("Unit Price (USD)")

    # Chart 3: Bar chart (RFM segments)
    seg_counts = rfm["Segment"].value_counts().reindex(["VIP", "Loyal", "New", "Churned"])
    sns.barplot(x=seg_counts.index, y=seg_counts.values, ax=axes[1, 0], color="#4F46E5")
    axes[1, 0].set_title("3. Bar Chart: Customer Count by RFM Segment")
    axes[1, 0].set_xlabel("RFM Segment")
    axes[1, 0].set_ylabel("Number of Customers")

    # Chart 4: Heatmap (Correlation matrix)
    sns.heatmap(stats["corr_matrix"], annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axes[1, 1])
    axes[1, 1].set_title("4. Heatmap: Numeric Feature Correlation Matrix")
    axes[1, 1].set_xlabel("Features")
    axes[1, 1].set_ylabel("Features")

    # Chart 5: Scatter plot (img_mean vs img_std)
    sns.scatterplot(data=df_clean, x="img_mean", y="img_std", hue="category", alpha=0.65, ax=axes[2, 0])
    axes[2, 0].set_title("5. Scatter Plot: Image Mean vs Image Std by Category")
    axes[2, 0].set_xlabel("Image Pixel Mean")
    axes[2, 0].set_ylabel("Image Pixel Std")

    # Chart 6: Line chart (Monthly revenue trend)
    monthly_sales = (
        df_clean.assign(month=df_clean["order_date"].dt.to_period("M").astype(str))
        .groupby("month")["amount"]
        .sum()
    )
    axes[2, 1].plot(monthly_sales.index, monthly_sales.values, marker="o", linewidth=2, color="#0EA5E9")
    axes[2, 1].set_title("6. Line Chart: Monthly Revenue Trend (2025)")
    axes[2, 1].set_xlabel("Month")
    axes[2, 1].set_ylabel("Total Revenue (USD)")
    axes[2, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    eda_file = out_path / "eda_6_charts.png"
    fig.savefig(eda_file, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return boxplot_file, eda_file


if __name__ == "__main__":
    bp_path, eda_path = generate_eda_charts()
    print(f"Saved: {bp_path}")
    print(f"Saved: {eda_path}")
