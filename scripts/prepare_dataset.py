"""Prepare a multimodal fashion e-commerce dataset with matching product images.

Source catalog & images:
    ashraq/fashion-product-images-small (Kaggle Fashion Product Images dataset)
Order transaction fields (customer_id, order_date, unit_price, quantity, amount):
    Synthesized per product row to enable RFM segmentation and time-series EDA.
"""
from pathlib import Path
import io
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.image as mpimg

PARQUET_PATH = Path("data/raw/fashion_products_part0.parquet")
OUT_DIR = Path("data")
TARGET_ROWS = 2000


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    pf = pq.ParquetFile(PARQUET_PATH)
    dfs = []
    # Read first few row groups (~3,000+ rows) to filter consistent (80, 60, 3) RGB images
    for rg_idx in range(min(4, pf.num_row_groups)):
        dfs.append(pf.read_row_group(rg_idx).to_pandas())
    raw_df = pd.concat(dfs, ignore_index=True)

    records = []
    images_list = []

    for _, row in raw_df.iterrows():
        img_dict = row["image"]
        if not isinstance(img_dict, dict) or not img_dict.get("bytes"):
            continue
        try:
            arr = mpimg.imread(io.BytesIO(img_dict["bytes"]), format="jpg")
        except Exception:
            continue

        # Normalize grayscale (80, 60) to (80, 60, 3) if needed, skip non-(80, 60)
        if arr.ndim == 2 and arr.shape == (80, 60):
            arr = np.stack([arr, arr, arr], axis=-1)
        elif arr.shape != (80, 60, 3):
            continue

        records.append(
            {
                "product_id": int(row["id"]),
                "product_name": row["productDisplayName"],
                "gender": row["gender"],
                "category": row["masterCategory"],
                "sub_category": row["subCategory"],
                "base_colour": row["baseColour"],
                "season": row["season"],
                "year": row["year"],
            }
        )
        images_list.append(arr.astype(np.uint8))

        if len(records) >= TARGET_ROWS:
            break

    df = pd.DataFrame(records)
    images = np.stack(images_list, axis=0)

    # Category-aware base pricing (USD)
    base_price_map = {
        "Apparel": (25.0, 12.0),
        "Footwear": (55.0, 22.0),
        "Accessories": (35.0, 18.0),
        "Personal Care": (18.0, 8.0),
        "Free Items": (10.0, 4.0),
    }
    unit_prices = np.empty(len(df), dtype=float)
    for i, cat in enumerate(df["category"]):
        mean_p, std_p = base_price_map.get(cat, (30.0, 10.0))
        unit_prices[i] = np.round(max(5.0, rng.normal(mean_p, std_p)), 2)

    quantities = rng.integers(1, 5, size=len(df))

    # Assign 350 customers with distinct behavioral tiers for natural RFM segmentation
    customer_ids = np.array([f"CUST_{i:04d}" for i in range(1, 351)])
    weights = rng.dirichlet(np.ones(len(customer_ids)) * 0.8)
    assigned_customers = rng.choice(customer_ids, size=len(df), p=weights)

    # Spread order dates across 2025-01-01 to 2025-12-31
    start_ts = pd.Timestamp("2025-01-01").value // 10**9
    end_ts = pd.Timestamp("2025-12-31 23:59:59").value // 10**9
    order_timestamps = rng.integers(start_ts, end_ts, size=len(df))
    order_dates = pd.to_datetime(order_timestamps, unit="s")

    df["customer_id"] = assigned_customers
    df["order_date"] = order_dates
    df["unit_price"] = unit_prices
    df["quantity"] = quantities
    df["amount"] = np.round(df["unit_price"] * df["quantity"], 2)
    df["image_id"] = np.arange(len(df))

    # Inject controlled outliers (~1.5%) in unit_price and quantity for IQR outlier analysis
    outlier_idx = rng.choice(df.index, size=30, replace=False)
    df.loc[outlier_idx[:15], "unit_price"] *= 8.0
    df.loc[outlier_idx[15:], "quantity"] = rng.integers(25, 60, size=15)
    df["amount"] = np.round(df["unit_price"] * df["quantity"], 2)

    # Inject controlled missing values in unit_price (~2.5%) for GroupBy imputation
    null_price_idx = rng.choice(df.index, size=50, replace=False)
    df.loc[null_price_idx, "unit_price"] = np.nan
    df.loc[null_price_idx, "amount"] = np.nan

    csv_path = OUT_DIR / "ecommerce_transactions.csv"
    npy_path = OUT_DIR / "product_images.npy"
    df.to_csv(csv_path, index=False)
    np.save(npy_path, images)

    print(f"Saved {csv_path} ({len(df)} rows, {len(df.columns)} cols)")
    print(f"Saved {npy_path} (shape={images.shape}, dtype={images.dtype})")
    print("Categories:\n", df["category"].value_counts())
    print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
    print("Unique customers:", df["customer_id"].nunique())


if __name__ == "__main__":
    main()
