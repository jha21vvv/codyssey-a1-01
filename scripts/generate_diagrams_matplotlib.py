"""Generate high-resolution architecture and flow diagrams using Matplotlib."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


OUT_DIR = Path("diagrams")


def _draw_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str,
    facecolor: str,
    edgecolor: str = "#1E293B",
) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.8,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h * 0.63,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        x + w / 2,
        y + h * 0.30,
        subtitle,
        ha="center",
        va="center",
        fontsize=8.8,
        color="#334155",
    )


def _draw_arrow(ax: plt.Axes, x1: float, y1: float, x2: float, y2: float) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>,head_length=7,head_width=4",
        color="#475569",
        linewidth=1.8,
    )
    ax.add_patch(arrow)


def build_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.93,
        "DataAnalyzer OOP Pipeline Architecture (src/pipeline.py)",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
    )

    _draw_box(ax, 0.04, 0.62, 0.20, 0.19, "1. Input Data", "CSV (2000x14)\n+ NPY (2000,80,60,3)", "#DBEAFE", "#2563EB")
    _draw_box(ax, 0.29, 0.62, 0.20, 0.19, "2. load_data()", "Parse order_date\nValidate row alignment", "#E0E7FF", "#4F46E5")
    _draw_box(ax, 0.54, 0.62, 0.20, 0.19, "3. Multimodal Feat", "NumPy Vector Mean/Std\nText Length/Words", "#DCFCE7", "#16A34A")
    _draw_box(ax, 0.78, 0.62, 0.19, 0.19, "4. Missing Impute", "GroupBy(category)\nMedian + Global Fallback", "#FEF3C7", "#D97706")

    _draw_arrow(ax, 0.24, 0.715, 0.29, 0.715)
    _draw_arrow(ax, 0.49, 0.715, 0.54, 0.715)
    _draw_arrow(ax, 0.74, 0.715, 0.78, 0.715)
    _draw_arrow(ax, 0.875, 0.62, 0.875, 0.43)

    _draw_box(ax, 0.72, 0.22, 0.25, 0.20, "5. detect_outliers()", "IQR = Q3 - Q1\nCap/Remove & Recompute", "#FEE2E2", "#DC2626")
    _draw_box(ax, 0.40, 0.22, 0.25, 0.20, "6. compute_statistics()", "Mean, Median, Std, Q1-Q3\nCorrelation Matrix", "#F3E8FF", "#9333EA")
    _draw_box(ax, 0.06, 0.22, 0.27, 0.20, "7. calculate_rfm()", "R/F/M Quantile Scores\nVIP / Loyal / New / Churned", "#CCFBF1", "#0D9488")

    _draw_arrow(ax, 0.72, 0.32, 0.65, 0.32)
    _draw_arrow(ax, 0.40, 0.32, 0.33, 0.32)

    ax.text(
        0.5,
        0.07,
        "Constraints: NumPy + Pandas + Matplotlib + Seaborn Only (No OpenCV / Pillow / Scikit-learn / Auto-EDA)",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        color="#475569",
    )

    fig.savefig(OUT_DIR / "01_pipeline_architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_vectorization_iqr_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.92,
        "NumPy Image Vectorization & IQR Outlier Capping Flow",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
    )

    _draw_box(ax, 0.05, 0.56, 0.26, 0.24, "4D Tensor (N, H, W, C)", "(2000, 80, 60, 3) uint8\nZero Python for-loops", "#DBEAFE", "#2563EB")
    _draw_box(ax, 0.37, 0.56, 0.27, 0.24, "Axis Reduction", "images.mean(axis=(1,2,3))\nimages.std(axis=(1,2,3))", "#DCFCE7", "#16A34A")
    _draw_box(ax, 0.70, 0.56, 0.25, 0.24, "Slicing Downsample", "images[:, ::2, ::2, :]\n-> (2000, 40, 30, 3)", "#E0E7FF", "#4F46E5")

    _draw_arrow(ax, 0.31, 0.68, 0.37, 0.68)
    _draw_arrow(ax, 0.64, 0.68, 0.70, 0.68)

    _draw_box(ax, 0.05, 0.14, 0.26, 0.24, "Raw Distribution", "unit_price / quantity\nContains 8x & bulk spikes", "#FEE2E2", "#DC2626")
    _draw_box(ax, 0.37, 0.14, 0.27, 0.24, "IQR Bounds", "IQR = Q3 - Q1\n[Q1 - 1.5*IQR, Q3 + 1.5*IQR]", "#FEF3C7", "#D97706")
    _draw_box(ax, 0.70, 0.14, 0.25, 0.24, "Winsorized / Clean", "series.clip(lower, upper)\nRecompute amount = P * Q", "#CCFBF1", "#0D9488")

    _draw_arrow(ax, 0.31, 0.26, 0.37, 0.26)
    _draw_arrow(ax, 0.64, 0.26, 0.70, 0.26)

    fig.savefig(OUT_DIR / "02_iqr_and_vectorization.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_rfm_diagram() -> None:
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.92,
        "RFM Customer Segmentation & Decision Tree",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="#0F172A",
    )

    _draw_box(ax, 0.04, 0.56, 0.25, 0.24, "Customer Aggregation", "GroupBy(customer_id)\nR: days, F: count, M: sum", "#DBEAFE", "#2563EB")
    _draw_box(ax, 0.36, 0.56, 0.28, 0.24, "Tie-Safe Quantile Score", "rank(method='first')\npd.qcut(..., q=4, [1,2,3,4])", "#E0E7FF", "#4F46E5")
    _draw_box(ax, 0.71, 0.56, 0.25, 0.24, "Segment Classifier", "Rule-based 4-tier mapping\nfrom (R, F, M) scores", "#F3E8FF", "#9333EA")

    _draw_arrow(ax, 0.29, 0.68, 0.36, 0.68)
    _draw_arrow(ax, 0.64, 0.68, 0.71, 0.68)

    _draw_box(ax, 0.04, 0.14, 0.21, 0.24, "VIP (R>=3,F>=3,M>=3)", "91 customers (32.5%)\nAvg $1,212.75 | 14.5 orders", "#DCFCE7", "#16A34A")
    _draw_box(ax, 0.28, 0.14, 0.21, 0.24, "Loyal (F>=3 or R,M>=3)", "56 customers (20.0%)\nAvg $561.10 | 7.1 orders", "#CCFBF1", "#0D9488")
    _draw_box(ax, 0.52, 0.14, 0.21, 0.24, "New (R>=3, F<=2)", "29 customers (10.4%)\nAvg $169.06 | 2.4 orders", "#FEF3C7", "#D97706")
    _draw_box(ax, 0.76, 0.14, 0.21, 0.24, "Churned (Others)", "104 customers (37.1%)\nAvg Recency 181.4 days", "#FEE2E2", "#DC2626")

    for tx in (0.145, 0.385, 0.625, 0.865):
        _draw_arrow(ax, 0.835, 0.56, tx, 0.38)

    fig.savefig(OUT_DIR / "03_rfm_segmentation_flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_pipeline_diagram()
    build_vectorization_iqr_diagram()
    build_rfm_diagram()
    print("Generated 3 diagrams in diagrams/")


if __name__ == "__main__":
    main()
