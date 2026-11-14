"""Object-oriented multimodal e-commerce data analysis pipeline.

Uses only NumPy and Pandas (no OpenCV, Pillow, scikit-learn, NLTK, or auto-EDA).
"""
# [1차]: 파이썬 3.8+ 환경에서 타입 힌트의 전방 참조를 허용하기 위해 __future__.annotations를 임포트합니다.
# [2차]: 설계도(타입 힌트)에 아직 아래쪽에 나올 클래스 이름을 먼저 적어두어도 에러가 나지 않게 미리 약속하는 기능입니다.
from __future__ import annotations

# [1차]: 파일 경로 객체(Path), 타입 힌트(Any, Optional), 문자열 리터럴 파싱(ast), 수치/표 처리(numpy, pandas) 모듈을 불러옵니다.
# [2차]: 요리에 필요한 기본 도마(Path), 영수증 번역기(ast), 고속 계산기(numpy), 엑셀 장부(pandas)를 주방에 꺼내놓습니다.
from pathlib import Path
from typing import Any, Optional
import ast
import numpy as np
import pandas as pd


# [1차]: 이커머스 멀티모달 전처리, 통계, RFM 세분화를 캡슐화한 DataAnalyzer 클래스를 정의합니다.
# [2차]: 원재료를 넣으면 다듬기, 이상치 제거, 성적표 계산까지 순서대로 처리해 주는 전용 분석 로봇의 설계도입니다.
class DataAnalyzer:
    """Multimodal e-commerce data preprocessing, EDA, and RFM segmentation pipeline."""

    # [1차]: 생성자 메서드(__init__)로 CSV/이미지 파일 경로와 내부 상태 속성(df, images, rfm_df, outlier_report)을 초기화합니다.
    # [2차]: 로봇의 전원을 켜면서 재료 창고 주소(경로)를 입력하고, 아직 비어 있는 작업대(df, images)를 준비하는 단계입니다.
    def __init__(
        self,
        csv_path: str | Path,
        images_path: Optional[str | Path] = None,
    ) -> None:
        # [1차]: 문자열 또는 Path 입력을 일관된 pathlib.Path 객체로 변환해 저장합니다.
        # [2차]: 손으로 적은 주소든 규격 주소든 모두 동일한 표준 우편번호 객체로 통일해 둡니다.
        self.csv_path = Path(csv_path)
        # [1차]: 이미지 파일 경로가 전달된 경우에만 Path 객체로 변환하고, 없으면 None을 유지합니다.
        # [2차]: 사진 상자가 같이 왔을 때만 주소를 적고, 없으면 빈칸(None)으로 남겨둡니다.
        self.images_path = Path(images_path) if images_path is not None else None
        # [1차]: 로드될 DataFrame, NumPy 이미지 배열, RFM 결과표, 이상치 보고서 딕셔너리를 초기화합니다.
        # [2차]: 빈 그릇 4개(표 데이터, 사진 묶음, 단골 등급표, 이상치 메모장)를 작업대 위에 올려놓습니다.
        self.df: Optional[pd.DataFrame] = None
        self.images: Optional[np.ndarray] = None
        self.rfm_df: Optional[pd.DataFrame] = None
        self.outlier_report: dict[str, Any] = {}

    # [1차]: 내부 헬퍼 메서드로 self.df가 로드되었는지 검증하고, 미로드 시 ValueError를 발생시킵니다.
    # [2차]: 요리를 시작하기 전에 냉장고에서 재료가 실제로 꺼내져 있는지 확인하는 안전 점검 버튼입니다.
    def _ensure_loaded(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("데이터가 로드되지 않았습니다. 먼저 load_data()를 호출하세요.")
        return self.df

    # [1차]: 정적 메서드로 문자열로 직렬화된 이미지 배열을 ast.literal_eval 또는 np.fromstring으로 복원합니다.
    # [2차]: 글자로 납작하게 눌러서 저장해 둔 사진 데이터를 다시 원래의 입체 숫자 격자(배열)로 되살리는 복원기입니다.
    @staticmethod
    def parse_image_array(raw_value: Any, shape: Optional[tuple[int, ...]] = None) -> np.ndarray:
        """Recover a NumPy array from a serialized CSV string using fromstring or literal_eval."""
        # [1차]: 이미 NumPy ndarray 타입이면 변환 없이 그대로 반환합니다.
        # [2차]: 이미 완제품 블록으로 되어 있으면 굳이 다시 조립하지 않고 바로 넘깁니다.
        if isinstance(raw_value, np.ndarray):
            return raw_value
        # [1차]: 문자열인 경우 대괄호('[') 시작 여부에 따라 파이썬 리스트 파싱(literal_eval) 또는 공백 구분자 파싱(fromstring)을 수행합니다.
        # [2차]: 대괄호로 묶인 상자 형태면 통째로 뜯고, 숫자만 나열되어 있으면 띄어쓰기 간격으로 하나씩 주워 담습니다.
        if isinstance(raw_value, str):
            text = raw_value.strip()
            if text.startswith("["):
                arr = np.array(ast.literal_eval(text), dtype=np.uint8)
            else:
                arr = np.fromstring(text, sep=" ", dtype=np.uint8)
            # [1차]: 목표 shape이 지정되어 있으면 reshape으로 차원을 복원해 반환합니다.
            # [2차]: 한 줄로 늘어선 숫자들을 원래 사진의 가로·세로 액자 틀에 맞춰 끼워 넣습니다.
            return arr.reshape(shape) if shape is not None else arr
        # [1차]: 배열도 문자열도 아닌 경우 TypeError 예외를 발생시킵니다.
        # [2차]: 정체불명의 포장지가 들어오면 작업을 멈추고 잘못된 형식이라고 알립니다.
        raise TypeError(f"지원하지 않는 이미지 배열 형식입니다: {type(raw_value)}")

    # [1차]: CSV 파일과 선택적 .npy 이미지 배열을 디스크에서 읽어와 날짜 파싱 및 행 개수 일치를 검증합니다.
    # [2차]: 장부(CSV)와 사진첩(NPY)을 동시에 펼쳐서 주문 날짜를 달력 형식으로 맞추고, 장부 줄 수와 사진 장수가 딱 맞는지 셉니다.
    def load_data(self) -> tuple[pd.DataFrame, Optional[np.ndarray]]:
        """Load tabular CSV data and optional NumPy image array."""
        # [1차]: CSV 파일 존재 여부를 확인하고 없으면 FileNotFoundError를 발생시킵니다.
        # [2차]: 창고에 장부 파일이 아예 없으면 작업을 시작하기 전에 바로 알려줍니다.
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {self.csv_path}")

        # [1차]: pandas.read_csv로 표 데이터를 읽고, order_date 컬럼이 있으면 datetime64 타입으로 변환합니다.
        # [2차]: 엑셀 표를 읽어온 뒤, 글자로 적힌 주문 날짜를 계산 가능한 진짜 달력 날짜로 바꿉니다.
        df = pd.read_csv(self.csv_path)
        if "order_date" in df.columns:
            df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

        # [1차]: 이미지 경로가 설정된 경우 존재 여부를 확인하고 np.load()로 배열을 로드한 뒤 행 수 일치를 검증합니다.
        # [2차]: 사진첩 파일이 있으면 꺼내와서 장부의 상품 개수(2,000개)와 사진 장수(2,000장)가 1:1로 맞는지 확인합니다.
        images: Optional[np.ndarray] = None
        if self.images_path is not None:
            if not self.images_path.exists():
                raise FileNotFoundError(f"이미지 배열 파일을 찾을 수 없습니다: {self.images_path}")
            images = np.load(self.images_path)
            if len(images) != len(df):
                raise ValueError(
                    f"데이터 행 수({len(df)})와 이미지 수({len(images)})가 일치하지 않습니다."
                )

        # [1차]: 검증이 완료된 DataFrame과 NumPy 배열을 인스턴스 속성에 저장하고 튜플로 반환합니다.
        # [2차]: 검수 끝난 장부와 사진첩을 작업대에 고정해 두고 결과물을 돌려줍니다.
        self.df = df
        self.images = images
        return self.df, self.images

    # [1차]: 반복문 없이 NumPy 축 연산과 슬라이싱으로 이미지 통계 피처를 추출하고, 텍스트 길이/단어 수 피처를 생성합니다.
    # [2차]: 사진 2,000장을 한 번에 스캐너에 올려 밝기 평균·편차와 축소판 평균을 뽑고, 상품명 글자 수와 단어 수도 함께 적어둡니다.
    def extract_multimodal_features(
        self,
        text_col: str = "product_name",
        downsample_step: int = 2,
    ) -> pd.DataFrame:
        """Extract vectorized image statistics (no Python loops) and text features."""
        # [1차]: 데이터가 로드되었는지 확인하고 원본 훼손을 막기 위해 DataFrame 복사본을 생성합니다.
        # [2차]: 작업대에 재료가 있는지 확인한 뒤, 원본 장부 대신 사본을 꺼내 새 컬럼을 적습니다.
        df = self._ensure_loaded().copy()

        # [1차]: 다운샘플링 보폭(stride)이 1 미만이면 ValueError를 발생시킵니다.
        # [2차]: 사진을 몇 칸씩 건너뛰며 축소할지 정할 때 0칸 이하로 건너뛰는 건 불가능하므로 막아줍니다.
        if downsample_step < 1:
            raise ValueError("downsample_step은 1 이상의 정수여야 합니다.")

        # [1차]: 텍스트 컬럼이 존재하면 결측치를 빈 문자열로 채운 뒤 벡터화된 문자열 길이(.str.len)와 단어 수(.str.split().str.len)를 계산합니다.
        # [2차]: 상품명 칸이 비어 있으면 빈칸으로 처리한 뒤, 이름 글자 수와 띄어쓰기 기준 단어 개수를 한 번에 셉니다.
        if text_col in df.columns:
            text_series = df[text_col].fillna("").astype(str)
            df["name_length"] = text_series.str.len()
            df["word_count"] = text_series.str.split().str.len()

        # [1차]: 이미지 배열이 존재할 때 빈 배열은 NaN으로 처리하고, 비어있지 않으면 배치 축(0)을 제외한 나머지 축에 대해 mean/std를 벡터 연산합니다.
        # [2차]: 사진이 0장이면 빈값으로 두고, 사진이 있으면 첫 번째 축(사진 번호)만 빼고 나머지 픽셀 전체의 평균 밝기와 명암 편차를 한 방에 계산합니다.
        if self.images is not None:
            if len(self.images) == 0:
                df["img_mean"] = np.nan
                df["img_std"] = np.nan
                df["img_downsampled_mean"] = np.nan
            else:
                img_axes = tuple(range(1, self.images.ndim))
                df["img_mean"] = self.images.mean(axis=img_axes)
                df["img_std"] = self.images.std(axis=img_axes)

                # [1차]: 4D(N,H,W,C) 또는 3D(N,H,W) 텐서에 대해 NumPy 슬라이싱(::downsample_step)으로 단순 다운샘플링을 수행합니다.
                # [2차]: 바둑판에서 가로·세로 한 칸씩 건너뛰며 점만 찍어 저해상도 썸네일을 만든 뒤 그 평균 밝기를 계산합니다.
                if self.images.ndim == 4:
                    downsampled = self.images[:, ::downsample_step, ::downsample_step, :]
                elif self.images.ndim == 3:
                    downsampled = self.images[:, ::downsample_step, ::downsample_step]
                else:
                    downsampled = self.images

                ds_axes = tuple(range(1, downsampled.ndim))
                df["img_downsampled_mean"] = downsampled.mean(axis=ds_axes)

        # [1차]: 갱신된 DataFrame을 self.df에 저장하고 반환합니다.
        # [2차]: 새 피처가 적힌 장부를 작업대에 다시 올려두고 결과를 반환합니다.
        self.df = df
        return self.df

    # [1차]: 지정한 그룹 컬럼(category 등)의 중앙값 또는 평균으로 결측치를 대치하고, 그룹 전체가 NaN이면 전체 통계로 2차 대치합니다.
    # [2차]: 가격표가 떨어진 상품이 있으면 "같은 코너(카테고리)의 중간 가격"으로 채우고, 코너 전체가 비었을 때만 매장 전체 중간값으로 채웁니다.
    def handle_missing_values(
        self,
        group_col: str = "category",
        target_cols: Optional[list[str]] = None,
        strategy: str = "median",
    ) -> pd.DataFrame:
        """Impute missing values using GroupBy statistics with global fallback."""
        # [1차]: 로드 여부를 확인하고 복사본을 생성합니다.
        # [2차]: 작업대 확인 후 안전하게 사본 장부를 엽니다.
        df = self._ensure_loaded().copy()

        # [1차]: 대치 전략이 'median' 또는 'mean'이 아니면 ValueError를 발생시킵니다.
        # [2차]: 중간값이나 평균 외의 엉뚱한 채우기 방식이 들어오면 즉시 멈춥니다.
        if strategy not in {"median", "mean"}:
            raise ValueError("strategy는 'median' 또는 'mean'이어야 합니다.")

        # [1차]: 대상 컬럼이 생략되면 기본값으로 ['unit_price']를 사용합니다.
        # [2차]: 어느 칸을 채울지 안 알려주면 기본적으로 단가(unit_price) 칸을 채웁니다.
        if target_cols is None:
            target_cols = ["unit_price"]

        # [1차]: 각 대상 컬럼에 대해 groupby(group_col).transform(strategy)로 그룹별 통계값을 매핑해 fillna를 수행하고, 남은 NaN은 전체 통계로 채웁니다.
        # [2차]: 같은 카테고리끼리 묶어 계산한 대표값으로 빈칸을 메우고, 그래도 빈칸이 남으면 전체 대표값으로 마저 메웁니다.
        for col in target_cols:
            if col not in df.columns:
                continue
            if group_col in df.columns:
                group_stats = df.groupby(group_col)[col].transform(strategy)
                df[col] = df[col].fillna(group_stats)

            global_val = df[col].median() if strategy == "median" else df[col].mean()
            if pd.notna(global_val):
                df[col] = df[col].fillna(global_val)

        # [1차]: 단가·수량·총액 컬럼이 모두 존재하면 대치된 단가를 반영해 amount = round(unit_price * quantity, 2)를 재계산합니다.
        # [2차]: 단가 빈칸을 채웠으니 '단가 × 수량 = 결제금액'도 비어 있지 않게 다시 계산해 적어줍니다.
        if {"unit_price", "quantity", "amount"}.issubset(df.columns):
            df["amount"] = np.round(df["unit_price"] * df["quantity"], 2)

        self.df = df
        return self.df

    # [1차]: IQR(Q3 - Q1) 방식으로 이상치를 탐지하고, action('cap', 'remove', 'none')에 따라 처리한 뒤 전후 통계 리포트를 반환합니다.
    # [2차]: 중앙 50% 구간의 폭(IQR)을 기준으로 정상 울타리를 세우고, 울타리를 넘은 극단값을 울타리 경계로 눌러주거나(cap) 빼낸(remove) 뒤 전후 성적표를 비교합니다.
    def detect_outliers(
        self,
        column: str = "unit_price",
        iqr_multiplier: float = 1.5,
        action: str = "cap",
    ) -> dict[str, Any]:
        """Detect and optionally cap or remove outliers using the IQR method."""
        # [1차]: 로드 여부, 대상 컬럼 존재 여부, action/iqr_multiplier 유효성을 검증합니다.
        # [2차]: 검사할 항목 이름과 처리 방법, 울타리 배수가 올바른지 먼저 확인합니다.
        df = self._ensure_loaded().copy()
        if column not in df.columns:
            raise KeyError(f"컬럼을 찾을 수 없습니다: {column}")
        if action not in {"cap", "remove", "none"}:
            raise ValueError("action은 'cap', 'remove', 'none' 중 하나여야 합니다.")
        if iqr_multiplier <= 0:
            raise ValueError("iqr_multiplier는 0보다 커야 합니다.")

        # [1차]: 결측치를 제외한 시리즈가 비어 있으면 ValueError를 발생시킵니다.
        # [2차]: 유효한 숫자가 하나도 없으면 울타리를 계산할 수 없으니 오류를 알립니다.
        series = df[column].dropna()
        if series.empty:
            raise ValueError(f"'{column}' 컬럼에 유효한 수치 데이터가 없습니다.")

        # [1차]: 1사분위수(Q1), 3사분위수(Q3), IQR 및 상·하한 경계(lower_bound, upper_bound)를 계산합니다.
        # [2차]: 하위 25%(Q1)와 상위 25%(Q3) 사이의 간격(IQR)을 구한 뒤, 양쪽으로 1.5배씩 여유를 둔 정상 울타리를 칩니다.
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr

        # [1차]: 울타리를 벗어난 불리언 마스크를 생성하고, 처리 전 기술통계를 기록합니다.
        # [2차]: 울타리 밖에 있는 튀는 값들에 표시를 해두고, 손대기 전의 평균·최대·최소값을 메모해 둡니다.
        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        before_stats = {
            "count": int(df[column].count()),
            "mean": float(df[column].mean()),
            "std": float(df[column].std(ddof=1)) if df[column].count() > 1 else 0.0,
            "min": float(df[column].min()),
            "max": float(df[column].max()),
        }

        # [1차]: action이 'cap'이면 clip()으로 경계값으로 고정하고, 'remove'이면 이상치 행을 제거하며 self.images도 동기화합니다.
        # [2차]: 'cap'이면 울타리 밖 값을 울타리 선에 맞춰 눌러주고, 'remove'이면 장부 줄과 짝지어진 사진도 같이 빼서 개수를 맞춥니다.
        if action == "cap":
            df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        elif action == "remove":
            df = df.loc[~outlier_mask].reset_index(drop=True)
            if self.images is not None:
                self.images = self.images[~outlier_mask.to_numpy()]

        # [1차]: 단가나 수량이 변경되었으면 총액(amount)을 다시 계산해 일관성을 유지합니다.
        # [2차]: 단가나 수량이 바뀌었으니 결제금액(단가 × 수량)도 새 값으로 업데이트합니다.
        if {"unit_price", "quantity", "amount"}.issubset(df.columns) and column in {
            "unit_price",
            "quantity",
        }:
            df["amount"] = np.round(df["unit_price"] * df["quantity"], 2)

        # [1차]: 처리 후 통계를 집계하고, 전후 리포트를 self.outlier_report에 저장한 뒤 반환합니다.
        # [2차]: 정리 후의 평균·최대·최소값을 다시 적어 '정리 전 vs 후' 비교 보고서를 완성합니다.
        after_stats = {
            "count": int(df[column].count()),
            "mean": float(df[column].mean()) if df[column].count() > 0 else 0.0,
            "std": float(df[column].std(ddof=1)) if df[column].count() > 1 else 0.0,
            "min": float(df[column].min()) if df[column].count() > 0 else 0.0,
            "max": float(df[column].max()) if df[column].count() > 0 else 0.0,
        }

        self.df = df
        report = {
            "column": column,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "outlier_count": int(outlier_mask.sum()),
            "action": action,
            "before": before_stats,
            "after": after_stats,
        }
        self.outlier_report[column] = report
        return report

    # [1차]: 수치형 컬럼들에 대한 요약 통계(평균, 중앙값, 표준편차, Q1/Q2/Q3)와 피어슨 상관계수 행렬을 계산합니다.
    # [2차]: 숫자 컬럼들의 성적표(평균·중간값·퍼짐 정도)와 "서로 얼마나 같이 움직이는지"를 보여주는 상관관계 표를 만듭니다.
    def compute_statistics(
        self,
        numeric_cols: Optional[list[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        """Compute summary statistics (mean, median, std, quartiles) and correlation matrix."""
        # [1차]: 로드된 DataFrame에서 지정된 수치형 컬럼 또는 전체 수치형 컬럼을 선택합니다.
        # [2차]: 작업대의 장부에서 계산할 숫자 칸들만 골라냅니다.
        df = self._ensure_loaded()
        if numeric_cols is None:
            numeric_df = df.select_dtypes(include=[np.number])
        else:
            numeric_df = df[numeric_cols]

        if numeric_df.empty:
            raise ValueError("통계를 계산할 수치형 컬럼이 없습니다.")

        # [1차]: 요약 통계 DataFrame과 상관계수 행렬(corr)을 딕셔너리로 묶어 반환합니다.
        # [2차]: 기본 성적표(summary)와 짝꿍 관계표(corr_matrix)를 한 묶음으로 건네줍니다.
        summary = pd.DataFrame(
            {
                "mean": numeric_df.mean(),
                "median": numeric_df.median(),
                "std": numeric_df.std(),
                "q1": numeric_df.quantile(0.25),
                "q2": numeric_df.quantile(0.50),
                "q3": numeric_df.quantile(0.75),
            }
        )
        corr_matrix = numeric_df.corr()
        return {"summary": summary, "corr_matrix": corr_matrix}

    # [1차]: 고객별 R, F, M 점수(1~4점)를 조합해 VIP, Loyal, New, Churned 4개 세그먼트로 분류하는 정적 메서드입니다.
    # [2차]: 최근성(R)·빈도(F)·금액(M) 점수를 보고 "최우수 VIP, 충성 단골, 신규 고객, 이탈 우려" 4개 이름표를 붙이는 규칙입니다.
    @staticmethod
    def _assign_rfm_segment(row: pd.Series) -> str:
        r, f, m = int(row["R_score"]), int(row["F_score"]), int(row["M_score"])
        if r >= 3 and f >= 3 and m >= 3:
            return "VIP"
        if f >= 3 or (r >= 3 and m >= 3):
            return "Loyal"
        if r >= 3 and f <= 2:
            return "New"
        return "Churned"

    # [1차]: 고객별 Recency(기준일 대비 경과일), Frequency(구매 건수), Monetary(총 구매액)를 집계하고 사분위 점수 및 세그먼트를 산출합니다.
    # [2차]: 고객별로 "마지막 방문 후 며칠 지났나(R), 몇 번 샀나(F), 총 얼마 썼나(M)"를 계산해 4개 고객 등급으로 나눕니다.
    def calculate_rfm(
        self,
        reference_date: Optional[str | pd.Timestamp] = None,
        customer_col: str = "customer_id",
        date_col: str = "order_date",
        amount_col: str = "amount",
    ) -> pd.DataFrame:
        """Calculate Recency, Frequency, Monetary scores and classify customers into 4 segments."""
        # [1차]: 필수 RFM 컬럼 존재 여부 및 유효 거래 행 존재 여부를 검증합니다.
        # [2차]: 고객ID·주문일·결제금액 3가지 핵심 열이 장부에 제대로 있는지 먼저 확인합니다.
        df = self._ensure_loaded().copy()
        required = {customer_col, date_col, amount_col}
        missing_cols = required - set(df.columns)
        if missing_cols:
            raise KeyError(f"RFM 계산에 필요한 컬럼이 없습니다: {sorted(missing_cols)}")

        valid = df.dropna(subset=[customer_col, date_col, amount_col]).copy()
        if valid.empty:
            raise ValueError("RFM 계산에 사용할 유효한 거래 데이터가 없습니다.")

        # [1차]: 기준일(reference_date)이 없으면 데이터의 최종 거래일을 기준일로 설정합니다.
        # [2차]: 기준 날짜를 따로 안 주면 장부에 찍힌 가장 마지막 영업일을 오늘로 삼습니다.
        valid[date_col] = pd.to_datetime(valid[date_col])
        ref_dt = (
            pd.to_datetime(reference_date)
            if reference_date is not None
            else valid[date_col].max()
        )

        # [1차]: 고객별로 그룹화하여 Recency(경과일수), Frequency(건수), Monetary(합계)를 집계합니다.
        # [2차]: 고객 이름표를 기준으로 영수증을 모아서 '마지막 방문 경과일, 방문 횟수, 총 결제금액' 3가지를 합산합니다.
        rfm = (
            valid.groupby(customer_col)
            .agg(
                Recency=(date_col, lambda s: (ref_dt - s.max()).days),
                Frequency=(date_col, "count"),
                Monetary=(amount_col, "sum"),
            )
            .reset_index()
        )

        # [1차]: 동률 경계 충돌을 방지하기 위해 rank(method="first")를 적용한 뒤 pd.qcut(..., q=4)로 1~4점 사분위 점수를 부여합니다.
        # [2차]: 같은 점수 손님이 많아도 4등분 경계가 겹쳐 에러가 나지 않도록 순서대로 번호표를 매긴 뒤 1~4점을 줍니다.
        # Use rank(method="first") so quantile binning never fails on tied values
        rfm["R_score"] = pd.qcut(
            rfm["Recency"].rank(method="first", ascending=False),
            q=4,
            labels=[1, 2, 3, 4],
        ).astype(int)
        rfm["F_score"] = pd.qcut(
            rfm["Frequency"].rank(method="first", ascending=True),
            q=4,
            labels=[1, 2, 3, 4],
        ).astype(int)
        rfm["M_score"] = pd.qcut(
            rfm["Monetary"].rank(method="first", ascending=True),
            q=4,
            labels=[1, 2, 3, 4],
        ).astype(int)

        # [1차]: 총합 RFM_score와 세그먼트 라벨을 부여하고 self.rfm_df에 저장 후 반환합니다.
        # [2차]: R+F+M 총점을 적고 4개 고객 등급 이름표를 붙여 최종 단골 분석표를 완성합니다.
        rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]
        rfm["Segment"] = rfm.apply(self._assign_rfm_segment, axis=1)

        self.rfm_df = rfm
        return self.rfm_df
