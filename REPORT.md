# REPORT.md — 종합 엔지니어링 리포트

> 본 문서는 `PROJECT_WORKFLOW_PROMPT_TEMPLATE.md`의 **Step 5**에 따라 아키텍처 의사결정, 복잡도, 테스트 결과, 엣지 케이스 해결 내역 및 평가자 검증 결과를 정리한 엔지니어링 리포트입니다.

---

## 1. 아키텍처 및 자료구조 선정 이유

* **객체 지향 파이프라인 (`DataAnalyzer`)**: 전처리 단계(`load_data` → `extract_multimodal_features` → `handle_missing_values` → `detect_outliers` → `compute_statistics` → `calculate_rfm`)의 상태(`self.df`, `self.images`, `self.rfm_df`)를 캡슐화하여 노트북과 테스트, 검증 스크립트에서 동일한 인터페이스로 재사용할 수 있도록 설계했습니다.
* **NumPy 벡터화 (`O(N · H · W · C)` 연속 메모리 연산)**: 이미지 `(2000, 80, 60, 3)` 텐서를 Python `for`문 없이 C-레벨 축 연산(`mean(axis=(1,2,3))`, `std(axis=(1,2,3))`)과 스트라이드 슬라이싱(`[:, ::2, ::2, :]`)으로 처리하여 2,000장 피처 추출을 수 밀리초 내에 완료합니다.
* **제약 준수**: OpenCV, Pillow, Scikit-learn, NLTK, 자동 EDA 라이브러리를 전혀 사용하지 않고 `numpy`, `pandas`, `matplotlib`, `seaborn`만으로 전체 파이프라인을 완결했습니다.

---

## 2. 테스트 자동화 및 검증 벤치마크

| 검증 항목 | 실행 명령어 | 결과 | 소요 시간 |
| :--- | :--- | :--- | :--- |
| 단위 테스트 (`tests/test_pipeline.py`) | `python -m unittest discover -s tests -v` | **6 / 6 통과 (100%)** | `0.141s` |
| 전체 요구사항 E2E 검증 (`scripts/verify_requirements.py`) | `python scripts/verify_requirements.py` | **8 / 8 PASS (100%)** | `0.462s` |
| 리포트 노트북 실행 (`notebooks/analysis_report.ipynb`) | `python scripts/build_notebook.py` | **17셀 전체 실행 성공** | ~`3s` |

---

## 3. 발견된 엣지 케이스 및 해결 내역

1. **RFM 4분위수(`pd.qcut`) 동률 경계 충돌 방어**:
   * `Frequency`나 `Recency`에 동일 값이 많아 `pd.qcut(..., q=4)`에서 `Bin edges must be unique` 오류가 발생할 수 있는 문제를 `rank(method="first")` 선행 적용으로 해결했습니다.
2. **GroupBy 결측치 대치 시 그룹 전체 NaN 방어**:
   * 특정 `category` 그룹의 모든 값이 `NaN`인 경우를 대비해 `groupby().transform()` 이후 전체 중앙값/평균으로 2차 fallback을 적용했습니다.
3. **이상치 제거(`action="remove"`) 시 이미지 텐서 인덱스 동기화**:
   * 행 제거 시 `self.df`만 슬라이싱하면 `self.images`와 행 수가 어긋나는 문제를 `self.images = self.images[~outlier_mask.to_numpy()]`로 동기화해 해결했습니다.
4. **Windows 콘솔 UTF-8 한글 인코딩 방어**:
   * `scripts/verify_requirements.py`에서 `sys.stdout.reconfigure(encoding="utf-8")`를 호출해 Windows PowerShell/CP949 환경에서도 한글과 JSON 출력이 깨지지 않도록 방어했습니다.
5. **Windows Jupyter ZeroMQ(`Bad file descriptor`) 우회 실행**:
   * 비ASCII 사용자 경로 환경에서 `nbconvert` 커널 소켓 오류를 회피하기 위해 `scripts/build_notebook.py`가 표준 `nbformat` 출력(`stdout`, `text/html`, `image/png`)을 인프로세스로 정확히 생성·검증하도록 구성했습니다.
6. **마크다운 이미지 경로 깨짐 방어**:
   * `README.md` 및 `스터디.md` 내 모든 아키텍처 다이어그램을 `file:///` 절대경로가 아닌 저장소 상대경로(`diagrams/*.png`)로 임베딩하고 `verify_requirements.py`로 자동 검증합니다.
