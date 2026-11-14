# REPORT.md — 종합 엔지니어링 리포트

> 본 문서는 `PROJECT_WORKFLOW_PROMPT_TEMPLATE.md`의 **Step 5**에 따라 아키텍처 의사결정, 복잡도, 테스트 결과, 통계적 한계·병목 분석, 엣지 케이스 해결 내역 및 평가자 검증 결과를 정리한 엔지니어링 리포트입니다.

---

## 1. 아키텍처 및 자료구조 선정 이유

* **객체 지향 파이프라인 (`DataAnalyzer`)**: 전처리 단계(`load_data` → `extract_multimodal_features` → `handle_missing_values`/`handle_missing` → `detect_outliers` → `compute_statistics` → `calculate_rfm`)의 상태(`self.df`, `self.images`, `self.rfm_df`)를 캡슐화하여 노트북(`notebooks/analysis_report.ipynb`), 시각화 모듈(`src/visualize.py`), 테스트, 검증 스크립트에서 동일한 인터페이스로 재사용할 수 있도록 설계했습니다.
* **NumPy 벡터화 (`O(N · H · W · C)` 연속 메모리 연산)**: 이미지 `(2000, 80, 60, 3)` 텐서를 Python `for`문 없이 C-레벨 축 연산(`mean(axis=(1,2,3))`, `std(axis=(1,2,3))`)과 스트라이드 슬라이싱(`[:, ::2, ::2, :]`)으로 처리하여 2,000장 피처 추출을 수 밀리초 내에 완료합니다.
* **제약 준수**: OpenCV, Pillow, Scikit-learn, NLTK, 자동 EDA 라이브러리를 전혀 사용하지 않고 `numpy`, `pandas`, `matplotlib`, `seaborn`만으로 전체 파이프라인을 완결했습니다.

---

## 2. 통계적·시스템적 한계 및 개선 전략

### 2.1 IQR의 정규분포 대비 장점 및 비대칭 분포 한계
* **장점(Robustness)**: 평균·표준편차 기반 Z-score는 극단값 자체가 평균과 표준편차를 왜곡하지만, IQR(`Q3 - Q1`)은 중앙값·사분위수 기반이므로 이상치에 강건합니다.
* **한계**: `amount`·`unit_price`처럼 오른쪽 꼬리가 긴 비대칭 분포에서는 정상 고가 주문까지 상한(`Q3 + 1.5*IQR`)을 넘어 이상치로 과잉 탐지될 수 있습니다. 필요 시 `np.log1p` 변환 후 IQR을 적용하거나 도메인 가격 상한을 병행해야 합니다.

### 2.2 GroupBy 결측치 대치의 통계적 위험
* **분산 축소(Variance Shrinkage)**: 그룹 내 결측을 모두 동일 대표값으로 채우면 실제 변동성이 사라져 그룹 내 분산과 표준편차가 과소추정됩니다.
* **그룹 간 차이 과장**: 결측 비율이 높은 그룹(예: `Personal Care`)일수록 단일 대치값으로 몰려 그룹 간 평균 차이가 과장되거나 변수 간 상관계수가 왜곡될 수 있습니다.

### 2.3 대규모 이미지 처리 병목 및 병렬화·배치·샘플링 개선안
* **병목 지점**: 대용량 이미지 처리 시 ① 개별 파일 랜덤 디스크 I/O, ② 4D 텐서 일괄 RAM 적재로 인한 메모리 부족(OOM), ③ 고해상도 픽셀 통계 연산 시 CPU 캐시 미스가 주요 병목입니다.
* **개선안**: `np.memmap`/청크 샤딩 기반 **배치 스트리밍**, `ProcessPoolExecutor`를 이용한 **병렬 디코딩/전처리**, 스트라이드 슬라이싱(`[:, ::n, ::n, :]`) 기반 **다운샘플링**으로 I/O와 메모리 사용량을 절감할 수 있습니다.

---

## 3. 테스트 자동화 및 실행 검증 벤치마크

| 검증 항목 | 실행 명령어 | 결과 |
| :--- | :--- | :--- |
| 단위 테스트 (`tests/test_pipeline.py`) | `python -m unittest discover -s tests -v` | **7 / 7 통과 (100%)** |
| 전체 요구사항 E2E 검증 (`scripts/verify_requirements.py`) | `python scripts/verify_requirements.py` | **8 / 8 PASS (100%)** |
| 리포트 노트북 실행 (`notebooks/analysis_report.ipynb`) | `python scripts/build_notebook.py` | **17셀 전체 실행 성공** |
| 시각화 6종 & 박스플롯 PNG 생성 (`src/visualize.py`) | `python src/visualize.py` | **2종 고해상도 PNG 저장 완료** |

---

## 4. 발견된 엣지 케이스 및 해결 내역

1. **메서드 명명 호환성 (`handle_missing` vs `handle_missing_values`)**:
   * 명세서 원문의 `handle_missing_values()`와 평가 기준의 `handle_missing()`을 모두 지원하도록 클래스 내 별칭(`handle_missing = handle_missing_values`)을 제공했습니다.
2. **RFM 4분위수(`pd.qcut`) 동률 경계 충돌 방어**:
   * `Frequency`나 `Recency`에 동일 값이 많아 `pd.qcut(..., q=4)`에서 `Bin edges must be unique` 오류가 발생할 수 있는 문제를 `rank(method="first")` 선행 적용으로 해결했습니다.
3. **GroupBy 결측치 대치 시 그룹 전체 NaN 방어**:
   * 특정 `category` 그룹의 모든 값이 `NaN`인 경우를 대비해 `groupby().transform()` 이후 전체 중앙값/평균으로 2차 fallback을 적용했습니다.
4. **이상치 제거(`action="remove"`) 시 이미지 텐서 인덱스 동기화**:
   * 행 제거 시 `self.df`만 슬라이싱하면 `self.images`와 행 수가 어긋나는 문제를 `self.images = self.images[~outlier_mask.to_numpy()]`로 동기화해 해결했습니다.
5. **Windows 콘솔 UTF-8 한글 인코딩 & Jupyter ZeroMQ 우회 실행**:
   * `sys.stdout.reconfigure(encoding="utf-8")`로 콘솔 인코딩을 방어하고, 비ASCII 경로 환경의 `nbconvert` 소켓 오류를 회피하기 위해 `scripts/build_notebook.py`가 표준 `nbformat` 출력을 인프로세스로 생성·검증하도록 구성했습니다.
6. **마크다운 이미지 경로 깨짐 방어**:
   * `README.md` 및 `스터디.md` 내 모든 시각화·다이어그램 이미지를 저장소 상대경로(`diagrams/*.png`)로 임베딩하고 `verify_requirements.py`로 자동 검증합니다.
