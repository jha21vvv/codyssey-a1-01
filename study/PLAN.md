# PLAN.md — 이커머스 멀티모달 데이터 분석 및 RFM 고객 세분화 계획서

> 본 계획서는 `PROJECT_WORKFLOW_PROMPT_TEMPLATE.md`의 **Step 1 (문제 분석 및 설계)**에 따라 `코디세이 a1-1.md`의 모든 기능 요구사항과 제약 사항을 분해하여 정리한 문서입니다.

---

## 1. 과제 목표 및 핵심 제약 사항

### 1.1 허용 / 금지 라이브러리
* **Python**: 3.8 이상 (현재 환경: Python 3.11.5)
* **허용 라이브러리**: `numpy`, `pandas`, `matplotlib`, `seaborn` (보너스 과제 3에 한해 `plotly` 예외 허용)
* **금지 라이브러리**:
  * 이미지 처리 전용 라이브러리 (`opencv-python`, `Pillow` 등) 사용 금지 → NumPy 벡터 연산 및 `matplotlib.image.imread`만 사용
  * 머신러닝/자연어처리 라이브러리 (`scikit-learn`, `nltk` 등) 사용 금지
  * 자동 EDA 라이브러리 (`pandas-profiling`, `sweetviz` 등) 사용 금지

### 1.2 이미지 처리 허용 범위
* 이미지 입력/변환: `matplotlib.image.imread()` 사용 가능, 반환되는 NumPy 배열 직접 조작
* 저장/복구: 문자열 배열 저장 시 `np.fromstring()` 또는 `ast.literal_eval()` 활용
* 리사이즈: NumPy 슬라이싱(`arr[:, ::n, ::n, :]` 또는 `arr[::n, ::n]`)으로 다운샘플링 구현

---

## 2. 데이터셋 규격 확인 (`data/`)
* `data/ecommerce_transactions.csv`: **2,000건 × 14개 컬럼** (최소 1,000건 / 8개 컬럼 이상 충족)
  * 수치형: `unit_price`, `quantity`, `amount`, `year`
  * 범주형: `gender`, `category`, `sub_category`, `base_colour`, `season`
  * 날짜형: `order_date` (`2025-01-01` ~ `2025-12-31`)
  * 텍스트: `product_name`
  * RFM 필수: `customer_id`, `order_date`, `amount`
* `data/product_images.npy`: **`(2000, 80, 60, 3)` `uint8`** (실제 상품 이미지와 1:1 매칭)

---

## 3. 모듈 및 기능 설계 (`src/pipeline.py`)

### 3.1 `DataAnalyzer` 클래스 설계 (OOP)
모든 분석 로직은 단일 책임 원칙(SRP)에 따라 메서드별로 분리하며, 주요 파라미터를 인자로 받아 재사용 가능하게 설계합니다.

| 메서드 | 역할 | 주요 파라미터 / 반환 |
| :--- | :--- | :--- |
| `__init__(self, csv_path, images_path=None)` | 데이터 경로 및 상태 초기화 | `self.df`, `self.images`, `self.rfm_df` |
| `load_data(self)` | CSV 및 `.npy` 이미지 배열 로드, `order_date` datetime 변환 | `(pd.DataFrame, np.ndarray)` |
| `extract_multimodal_features(self, downsample_step=2)` | **이미지**: `for`문 없이 NumPy 벡터 연산으로 평균(`img_mean`), 표준편차(`img_std`) 및 슬라이싱 다운샘플링 통계 추출<br>**텍스트**: `product_name` 단어 수(`word_count`), 문자열 길이(`name_length`) 파생 변수 생성 | `pd.DataFrame` |
| `handle_missing_values(self, group_col="category", target_cols=None, strategy="median")` | 결측치 현황 집계 후 `GroupBy` 통계(`median`/`mean`)로 `unit_price` 대치 및 `amount` 재계산 | `pd.DataFrame` |
| `detect_outliers(self, column="unit_price", iqr_multiplier=1.5, action="cap")` | IQR(`Q3 - Q1`) 기반 이상치 탐지 및 처리(`cap`/윈저라이징 또는 `remove`), 처리 전후 요약 통계 반환 | `dict` (before/after 통계, bounds, outlier_count) |
| `compute_statistics(self, numeric_cols=None)` | 평균, 중앙값, 표준편차, 사분위수(Q1, Q2, Q3) 및 상관계수 행렬 계산 | `dict` (`summary`, `corr_matrix`) |
| `calculate_rfm(self, reference_date=None, customer_col="customer_id", date_col="order_date", amount_col="amount")` | 기준일(기본값: 데이터 최종 거래일 + 1일 또는 최종 거래일) 대비 Recency, Frequency, Monetary 계산 및 4개 이상 세그먼트(`VIP`, `Loyal`, `New`, `At Risk / Churned`) 분류 | `pd.DataFrame` |

### 3.2 예외 및 엣지 케이스 방어 설계
* `load_data()` 호출 전 다른 메서드 실행 시 명확한 `ValueError` 발생
* `GroupBy` 대치 시 특정 그룹 전체가 `NaN`인 경우 전체 중앙값/평균으로 2차 fallback
* `detect_outliers()`에서 `IQR == 0`인 컬럼 방어 및 음수/0 값 검증
* `calculate_rfm()`에서 4분위수(`pd.qcut`) 동률(duplicate bin edges) 발생 시 `rank(method="first")`로 안전하게 점수화
* Windows 환경 UTF-8 콘솔 출력 방어 및 마크다운 이미지 상대경로(`diagrams/*.png`) 준수

---

## 4. 분석 리포트 노트북 설계 (`notebooks/analysis_report.ipynb`)
1. **데이터 로드 및 기본 탐색**: `info()`, `describe()`, `head()` 결과 + 수치 근거 1개 이상 및 시사점 1문장 이상 마크다운 해석
2. **멀티모달 피처 엔지니어링**: NumPy 벡터 연산 이미지 특성(`img_mean`, `img_std`, 다운샘플링) 및 텍스트 특성(`word_count`, `name_length`)
3. **결측치 대치 및 IQR 이상치 처리**: 처리 전후 비교 및 해석
4. **기술통계 및 상관분석**: 주요 수치형 변수 요약 통계 + 최소 2쌍 이상 상관계수 해석
5. **시각화 6종** (모든 차트에 제목·축 레이블 포함):
   * 히스토그램 (수치형 변수 분포, 예: `amount` 또는 `unit_price`)
   * 박스플롯 (이상치 처리 전후 비교)
   * 막대그래프 (범주형 `category` 빈도 또는 RFM 세그먼트별 고객 수)
   * 히트맵 (상관행렬 또는 RFM 세그먼트 분포)
   * 산점도 (두 수치형 변수 관계, 예: `img_mean` vs `unit_price` 또는 `quantity` vs `amount`)
   * 라인차트 (월별 매출 추이)
6. **RFM 기반 고객 세분화**: 최소 4개 세그먼트 특징 분석 및 비즈니스 인사이트 도출

---

## 5. README.md 비즈니스 인사이트 규격
최종 `README.md`에는 최소 3가지 비즈니스 인사이트를 수록하며, 각 인사이트마다 아래 3요소를 반드시 포함합니다.
1. **(근거)** 인사이트를 뒷받침하는 수치 또는 그래프 1개 이상
2. **(실행)** 대상 세그먼트 + 실행 방법 + 기대 효과
3. **(검증)** 추가로 필요한 데이터 또는 반증 가능성 1문장

---

## 6. 7단계 워크플로우 마일스톤 (`PROJECT_WORKFLOW_PROMPT_TEMPLATE.md`)
* **Step 1 (현재)**: 요구사항 분해 및 `PLAN.md` 작성 (사용자 승인 대기)
* **Step 2**: `src/pipeline.py` (`DataAnalyzer`) 및 `notebooks/analysis_report.ipynb` 구현·실행
* **Step 3**: `tests/test_pipeline.py` 초고속 단위 테스트 작성 (정상/예외/엣지케이스 100% 통과 검증)
* **Step 4**: `스터디.md` (과제 전문 + 1:1 매핑표 + 주니어 비유) 및 `scripts/generate_diagrams_matplotlib.py` (`diagrams/*.png` 생성 및 상대경로 임베딩)
* **Step 5**: `REPORT.md`, `scripts/verify_requirements.py` 작성 및 `README.md` 최종 완성 (인사이트 3가지 + 평가자 검증 섹션)
* **Step 6**: `scripts/build_notebooklm_source.py`, `NOTEBOOKLM_FULL_SOURCE.txt`, `NOTEBOOKLM_GUIDE.md` 작성
* **Step 7**: `src/pipeline.py` 전 라인 `[1차 기술] + [2차 비유]` 2단 주석 내재화 및 최종 테스트 검증

---

> **승인 요청**: 위 `PLAN.md` 설계대로 Step 2(모듈 및 노트북 구현)로 진행해도 될까요?
