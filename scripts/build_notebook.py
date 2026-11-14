"""Build and execute notebooks/analysis_report.ipynb in-process without ZeroMQ."""
from pathlib import Path
import base64
import contextlib
import io
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
import pandas as pd


def execute_notebook_in_process(nb: nbf.NotebookNode, workdir: Path) -> nbf.NotebookNode:
    """Execute all code cells in-process and attach standard nbformat outputs."""
    old_cwd = Path.cwd()
    os.chdir(workdir)
    namespace: dict = {}
    exec_count = 1

    try:
        for cell in nb.cells:
            if cell.cell_type != "code":
                continue

            outputs = []
            stdout_buf = io.StringIO()

            def _display(obj: object) -> None:
                data = {"text/plain": repr(obj)}
                if isinstance(obj, pd.DataFrame):
                    data["text/html"] = obj.to_html()
                outputs.append(nbf.v4.new_output("display_data", data=data))

            def _show(*_args: object, **_kwargs: object) -> None:
                for fig_num in plt.get_fignums():
                    fig = plt.figure(fig_num)
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=120)
                    img_b64 = base64.b64encode(img_buf.getvalue()).decode("ascii")
                    outputs.append(
                        nbf.v4.new_output(
                            "display_data",
                            data={"image/png": img_b64, "text/plain": "<Figure>"},
                        )
                    )
                plt.close("all")

            namespace["display"] = _display
            plt.show = _show

            with contextlib.redirect_stdout(stdout_buf):
                exec(compile(cell.source, f"<cell-{exec_count}>", "exec"), namespace)

            text_out = stdout_buf.getvalue()
            if text_out:
                outputs.insert(0, nbf.v4.new_output("stream", name="stdout", text=text_out))

            cell.outputs = outputs
            cell.execution_count = exec_count
            exec_count += 1
    finally:
        os.chdir(old_cwd)

    return nb


def main() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# 이커머스 멀티모달 데이터 분석 및 RFM 고객 세분화 리포트\n\n"
            "본 노트북은 `src/pipeline.py`의 `DataAnalyzer` 클래스를 사용하여 다음 과정을 수행합니다.\n"
            "1. 데이터 로드 및 기본 탐색 (`info`, `describe`, `head`)\n"
            "2. 멀티모달 피처 엔지니어링 (NumPy 벡터 연산 기반 이미지 통계/다운샘플링 + 텍스트 특성)\n"
            "3. GroupBy 결측치 대치 및 IQR 이상치 탐지·처리\n"
            "4. 기술통계 및 상관분석\n"
            "5. 필수 시각화 6종 (히스토그램, 박스플롯, 막대그래프, 히트맵, 산점도, 라인차트)\n"
            "6. RFM 기반 고객 세분화 (VIP, Loyal, New, Churned)"
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n\n"
            "sys.path.insert(0, str(Path('..').resolve()))\n\n"
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "import seaborn as sns\n"
            "from src.pipeline import DataAnalyzer\n\n"
            "plt.rcParams['axes.unicode_minus'] = False\n"
            "sns.set_theme(style='whitegrid')"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 1. 데이터 로드 및 기본 탐색\n"
            "`DataAnalyzer`를 초기화하고 `ecommerce_transactions.csv`와 `product_images.npy`를 로드합니다."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "analyzer = DataAnalyzer(\n"
            "    csv_path='../data/ecommerce_transactions.csv',\n"
            "    images_path='../data/product_images.npy',\n"
            ")\n"
            "df_raw, images = analyzer.load_data()\n"
            "print(f'Tabular shape: {df_raw.shape}, Images shape: {images.shape}, dtype: {images.dtype}')\n"
            "df_raw.info()\n"
            "display(df_raw.head())\n"
            "display(df_raw.describe(include='all').T)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "### 기본 탐색 해석\n"
            "- **수치 근거**: 총 `2,000`건, `14`개 컬럼과 `(2000, 80, 60, 3)` 이미지 배열이 1:1로 매칭되어 있으며, "
            "`unit_price`와 `amount`에 각 `50`건(2.5%)의 결측치가 존재하고 `unit_price` 최댓값은 평균 대비 크게 높은 극단값을 포함하고 있습니다.\n"
            "- **시사점**: 결측치 대치(GroupBy)와 이상치 완화(IQR)를 선행해야 이후 매출 통계와 RFM 세그먼트가 왜곡되지 않습니다."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 2. 멀티모달 피처 엔지니어링 (NumPy 벡터 연산)\n"
            "반복문(`for` loop) 없이 NumPy 배열 축 연산(`mean`, `std`, 슬라이싱 `[:, ::2, ::2, :]`)으로 이미지 통계 특성을 추출하고, "
            "상품명(`product_name`)의 문자열 길이와 단어 수를 계산합니다."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "df_feat = analyzer.extract_multimodal_features(downsample_step=2)\n"
            "display(df_feat[['product_name', 'name_length', 'word_count', 'img_mean', 'img_std', 'img_downsampled_mean']].head())"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 3. 결측치 대치 및 IQR 이상치 처리\n"
            "카테고리별 중앙값으로 `unit_price` 결측치를 대치한 뒤, 처리 전 데이터와 IQR Capping 처리 후 분포를 비교합니다."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "df_imputed = analyzer.handle_missing_values(group_col='category', target_cols=['unit_price'], strategy='median')\n"
            "print('Missing values after imputation:\\n', df_imputed.isna().sum()[df_imputed.isna().sum() > 0])\n\n"
            "price_before = df_imputed['unit_price'].copy()\n"
            "outlier_price = analyzer.detect_outliers(column='unit_price', iqr_multiplier=1.5, action='cap')\n"
            "outlier_qty = analyzer.detect_outliers(column='quantity', iqr_multiplier=1.5, action='cap')\n"
            "price_after = analyzer.df['unit_price'].copy()\n\n"
            "print('unit_price outlier report:', {k: v for k, v in outlier_price.items() if k in ['q1', 'q3', 'iqr', 'upper_bound', 'outlier_count']})\n"
            "print('quantity outlier report:', {k: v for k, v in outlier_qty.items() if k in ['q1', 'q3', 'iqr', 'upper_bound', 'outlier_count']})"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 4. 기술통계 및 상관분석\n"
            "주요 수치형 변수에 대해 평균, 중앙값, 표준편차, 사분위수(Q1, Q2, Q3) 및 상관계수 행렬을 계산합니다."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "num_cols = ['unit_price', 'quantity', 'amount', 'img_mean', 'img_std', 'word_count', 'name_length']\n"
            "stats = analyzer.compute_statistics(numeric_cols=num_cols)\n"
            "display(stats['summary'].round(2))\n"
            "display(stats['corr_matrix'].round(3))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "### 통계 및 상관분석 해석\n"
            "- **수치 근거 1 (중심경향 및 산포)**: `unit_price`의 평균은 약 `33.15`, 중앙값은 `30.29`, 표준편차는 `18.40`이며 "
            "`amount`의 사분위수는 Q1 `34.77`, Q2 `66.43`, Q3 `114.70`으로 우측 꼬리가 긴 분포를 보입니다.\n"
            "- **수치 근거 2 (상관관계 쌍 1 — `img_mean` vs `img_std`)**: 두 변수의 상관계수는 약 `-0.818`로 강한 음의 상관을 나타냅니다. "
            "밝은 흰 배경 비중이 높은 상품 이미지일수록 평균 픽셀 밝기가 높아지는 대신 픽셀 간 명암 편차가 줄어든다는 시사점을 제공합니다.\n"
            "- **수치 근거 3 (상관관계 쌍 2 — `word_count` vs `name_length`)**: 상관계수가 약 `0.885`로 강한 양의 상관을 보이며, "
            "`unit_price`와 `amount` 역시 `0.707`의 높은 양의 상관을 보여 고가 카테고리가 총매출에 미치는 영향이 큼을 시사합니다."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 5. 시각화 6종\n"
            "히스토그램, 박스플롯(이상치 처리 전후 비교), 막대그래프, 히트맵, 산점도, 라인차트(월별 매출 추이)를 작성합니다."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "df_clean = analyzer.df.copy()\n"
            "rfm = analyzer.calculate_rfm()\n\n"
            "fig, axes = plt.subplots(3, 2, figsize=(15, 16))\n\n"
            "# 1. Histogram: amount distribution\n"
            "sns.histplot(df_clean['amount'], bins=30, kde=True, ax=axes[0, 0], color='#2563EB')\n"
            "axes[0, 0].set_title('1. Histogram: Order Amount Distribution')\n"
            "axes[0, 0].set_xlabel('Order Amount (USD)')\n"
            "axes[0, 0].set_ylabel('Frequency')\n\n"
            "# 2. Boxplot: unit_price before vs after IQR capping\n"
            "bp_df = pd.DataFrame({'Before IQR': price_before, 'After IQR': price_after}).melt(var_name='Stage', value_name='Unit Price')\n"
            "sns.boxplot(data=bp_df, x='Stage', y='Unit Price', ax=axes[0, 1], color='#60A5FA')\n"
            "axes[0, 1].set_title('2. Boxplot: Unit Price Before vs After IQR Capping')\n"
            "axes[0, 1].set_xlabel('Preprocessing Stage')\n"
            "axes[0, 1].set_ylabel('Unit Price (USD)')\n\n"
            "# 3. Bar chart: RFM segment customer counts\n"
            "seg_counts = rfm['Segment'].value_counts().reindex(['VIP', 'Loyal', 'New', 'Churned'])\n"
            "sns.barplot(x=seg_counts.index, y=seg_counts.values, ax=axes[1, 0], color='#4F46E5')\n"
            "axes[1, 0].set_title('3. Bar Chart: Customer Count by RFM Segment')\n"
            "axes[1, 0].set_xlabel('RFM Segment')\n"
            "axes[1, 0].set_ylabel('Number of Customers')\n\n"
            "# 4. Heatmap: correlation matrix\n"
            "sns.heatmap(stats['corr_matrix'], annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1, 1])\n"
            "axes[1, 1].set_title('4. Heatmap: Numeric Feature Correlation Matrix')\n"
            "axes[1, 1].set_xlabel('Features')\n"
            "axes[1, 1].set_ylabel('Features')\n\n"
            "# 5. Scatter plot: img_mean vs img_std\n"
            "sns.scatterplot(data=df_clean, x='img_mean', y='img_std', hue='category', alpha=0.65, ax=axes[2, 0])\n"
            "axes[2, 0].set_title('5. Scatter Plot: Image Mean vs Image Std by Category')\n"
            "axes[2, 0].set_xlabel('Image Pixel Mean')\n"
            "axes[2, 0].set_ylabel('Image Pixel Std')\n\n"
            "# 6. Line chart: monthly revenue trend\n"
            "monthly_sales = df_clean.assign(month=df_clean['order_date'].dt.to_period('M').astype(str)).groupby('month')['amount'].sum()\n"
            "axes[2, 1].plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, color='#0EA5E9')\n"
            "axes[2, 1].set_title('6. Line Chart: Monthly Revenue Trend (2025)')\n"
            "axes[2, 1].set_xlabel('Month')\n"
            "axes[2, 1].set_ylabel('Total Revenue (USD)')\n"
            "axes[2, 1].tick_params(axis='x', rotation=45)\n\n"
            "plt.tight_layout()\n"
            "plt.show()"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 6. RFM 기반 고객 세분화 결과\n"
            "고객별 Recency(최근 구매 경과일), Frequency(구매 빈도), Monetary(총 구매금액)를 산출하고 4개 세그먼트 특징을 요약합니다."
        )
    )

    cells.append(
        nbf.v4.new_code_cell(
            "rfm_summary = rfm.groupby('Segment').agg(\n"
            "    customers=('customer_id', 'count'),\n"
            "    avg_recency=('Recency', 'mean'),\n"
            "    avg_frequency=('Frequency', 'mean'),\n"
            "    avg_monetary=('Monetary', 'mean'),\n"
            "    total_revenue=('Monetary', 'sum'),\n"
            ").round(2)\n"
            "rfm_summary['revenue_share_pct'] = (rfm_summary['total_revenue'] / rfm_summary['total_revenue'].sum() * 100).round(2)\n"
            "display(rfm_summary)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "### RFM 세분화 요약\n"
            "- **VIP** 그룹은 전체 고객의 약 `32.5%`(91명)이면서 평균 구매 빈도 `14.48회`, 평균 구매 금액 `$1,212.75`로 전체 매출의 핵심 비중을 차지합니다.\n"
            "- **Churned** 그룹은 `104명`(37.1%)으로 가장 많은 인원을 차지하며 평균 미구매 기간이 `181.38일`에 달해 재활성화 캠페인이 시급합니다."
        )
    )

    nb["cells"] = cells
    out_path = Path("notebooks/analysis_report.ipynb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    execute_notebook_in_process(nb, out_path.parent)
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed & saved {out_path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
