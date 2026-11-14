# 📋 평가 피드백 분석 및 개선 계획서 (`EVALUATION_IMPROVEMENT_GUIDE.md`)

> 자동 평가기 피드백(평가 항목 `#1`, `#2`, `#5`, `#6`, `#13`, `#14`, `#17`, `#18`, `#19`)을 원본 과제 명세서([코디세이 a1-1.md](코디세이%20a1-1.md))와 대조하여 **원인**과 **구체적 개선 방안**을 정리한 문서입니다.

---

## 1. 한눈에 보는 평가 항목 분류

| 평가 항목 | 현재 상태 | 명세서(`코디세이 a1-1.md`) 포함 여부 | 원인 요약 | 개선 방향 |
| :--- | :---: | :--- | :--- | :--- |
| **#1** `handle_missing` 이름 불일치 | FAIL | 원문은 `handle_missing_values()` | 평가기 루브릭이 `handle_missing()` 탐색 | `DataAnalyzer`에 `handle_missing = handle_missing_values` 별칭 추가 |
| **#2** 노트북 실행 증거 미제출 | FAIL | 간접 포함 (`analysis_report.ipynb`) | `.ipynb` 내부 출력 외 별도 실행 로그/스냅샷 미수록 | 실행 로그 및 차트 PNG 스냅샷을 `README.md`/`REPORT.md`에 수록 |
| **#5** 이상치 전후 박스플롯 누락 | FAIL | 포함 (기능 요구사항 4, 6) | 노트북에는 구현되어 있으나 `.py`/이미지 파싱 누락 | `src/visualize.py` 분리 + 전후 비교 박스플롯 PNG 생성 및 `README.md` 임베딩 |
| **#6** 시각화 6종 및 제목/축 레이블 누락 | FAIL | 포함 (기능 요구사항 6) | 노트북에는 구현되어 있으나 평가기가 `.ipynb` 미파싱 | `src/visualize.py`에 6종 시각화 구현 + `diagrams/eda_6_charts.png` 임베딩 |
| **#13** IQR 장점 및 비대칭 분포 한계 | FAIL | 원문 미기재 (심화 루브릭) | 이론적 장단점/한계 문서화 부재 | Z-score 대비 강건성(Robustness)과 왜도(Skewness) 한계를 `README.md`/`REPORT.md`에 기술 |
| **#14** GroupBy 대치의 통계적 위험 | FAIL | 원문 미기재 (심화 루브릭) | 분산 축소/그룹 간 과장 리스크 설명 부재 | 분산 축소(Variance Shrinkage) 및 그룹 차이 과장 예시를 `README.md`/`REPORT.md`에 기술 |
| **#17** 대규모 이미지 병목 및 병렬화 | FAIL | 원문 미기재 (심화 루브릭) | 디스크 I/O·메모리 병목 및 개선안 문서화 부재 | I/O·메모리 병목과 배치/멀티프로세싱/샘플링 개선안을 `README.md`/`REPORT.md`에 기술 |
| **#18** 우선 액션 KPI 기준 | PASS(보완) | 원문 미기재 (심화 루브릭) | A/B 테스트는 명시했으나 우선순위 KPI 미명시 | 매출 기여도·이탈 위험도·ROI 등 우선 KPI 기준을 `README.md`에 명시 |
| **#19** ML 확장 타겟 및 피처 목록 | FAIL | 원문 미기재 (심화 루브릭) | 후속 머신러닝 과제 연계 설계 부재 | 예측 타겟(`is_churned`, `LTV` 등)과 피처 목록(이미지·텍스트·RFM)을 `README.md`에 별도 섹션으로 추가 |

---

## 2. 항목별 상세 원인 및 개선 방안

### A. 코드 및 실행 증거 보완 (`#1`, `#2`, `#5`, `#6`)

1. **#1 메서드명 일치 (`src/pipeline.py`)**
   * **원인**: `코디세이 a1-1.md` 47행에는 `handle_missing_values()`로 표기되어 있으나, 평가 체커가 `handle_missing()`을 검사함.
   * **개선**: `DataAnalyzer` 클래스 내부에 `handle_missing = handle_missing_values` 별칭(alias)을 추가하고 `README.md`에 명명 규칙을 명시.

2. **#2 노트북 실행 증거 (`README.md`, `REPORT.md`)**
   * **원인**: `notebooks/analysis_report.ipynb`는 전체 셀이 실행되어 있으나, 평가기가 별도 텍스트 실행 로그 또는 결과 이미지 스냅샷을 요구함.
   * **개선**: `scripts/build_notebook.py` 실행 콘솔 로그와 생성된 시각화 PNG를 `README.md` 및 `REPORT.md`에 직접 첨부.

3. **#5 & #6 시각화 6종 및 전후 박스플롯 (`src/visualize.py`, `diagrams/`)**
   * **원인**: `notebooks/analysis_report.ipynb` 5절에 이미 6종 차트(히스토그램, 이상치 전후 박스플롯, 막대그래프, 히트맵, 산점도, 월별 라인차트)와 제목·축 레이블이 모두 구현되어 있으나, 자동 평가기가 `.ipynb` 내부 코드를 읽지 못함.
   * **개선**:
     * `src/visualize.py` 모듈을 신설하여 6종 시각화 및 이상치 전후 박스플롯 생성 함수를 코드로 제공.
     * `diagrams/eda_6_charts.png` 및 `diagrams/outlier_boxplot_comparison.png`로 저장 후 `README.md`에 상대경로로 임베딩.

---

### B. 통계·엔지니어링 심화 문서 보완 (`#13`, `#14`, `#17`, `#18`, `#19`)

4. **#13 IQR 이상치 탐지의 장점과 비대칭 분포 한계**
   * **장점(Robustness)**: 평균과 표준편차를 쓰는 Z-score와 달리 중앙값 기반 사분위수($Q_1, Q_3$)를 사용하므로 극단값 자체에 의해 경계가 왜곡되지 않음.
   * **한계(비대칭 분포)**: 이커머스 주문 금액(`amount`)이나 단가처럼 오른쪽 꼬리가 긴(Right-skewed) 분포에서는 정상적인 고가 주문까지 상한(`Q3 + 1.5*IQR`)을 넘어 이상치로 과잉 탐지할 수 있음. (필요 시 로그 변환 후 IQR 적용 또는 도메인 상한 병행 필요)

5. **#14 GroupBy 결측치 대치의 통계적 위험**
   * **분산 축소(Variance Shrinkage)**: 동일 카테고리(`category`) 내 결측치를 모두 같은 대표값(중앙값/평균)으로 채우면 실제 변동성이 사라져 표본 분산과 표준편차가 인위적으로 작아짐.
   * **그룹 간 차이 과장 & 상관관계 왜곡**: 결측 비율이 높은 카테고리일수록 대치값으로 값이 몰려 카테고리 간 평균 차이가 과장되거나, 다른 변수와의 상관계수가 왜곡될 수 있음. (예: `Personal Care` 그룹 결측을 모두 `$18.0`으로 대치 시 해당 그룹 내 가격 분산이 0에 가깝게 수렴)

6. **#17 대규모 이미지 처리 병목 지점 및 개선 방안**
   * **병목 지점**:
     1. **디스크 I/O**: 수십만 장의 개별 JPEG/PNG 파일을 매번 열어 디코딩할 때 랜덤 I/O 지연 발생.
     2. **메모리 적재**: 전체 이미지를 한 번에 4D `uint8`/`float32` 텐서로 RAM에 올리면 OOM(Out-Of-Memory) 위험.
     3. **전처리 연산**: 고해상도 이미지 전체 픽셀 통계 계산 시 CPU 캐시 미스 증가.
   * **개선 방안**:
     * `np.memmap` 또는 샤딩(`.npy`/Parquet 청크) 기반 **미니배치(Batch) 스트리밍 처리**
     * `concurrent.futures.ProcessPoolExecutor`를 활용한 **디코딩/통계 추출 병렬화**
     * 학습·탐색 단계에서 `arr[:, ::n, ::n, :]` **스트라이드 다운샘플링(Subsampling)**으로 메모리·연산량 $n^2$배 절감

7. **#18 우선 액션 세그먼트 선정 KPI 기준**
   * **선정 기준**: ① **매출 기여도(Revenue Share)**, ② **객단가 및 구매 빈도(ARPU / Frequency)**, ③ **이탈 위험도(Recency 경과일)**, ④ **캠페인 ROI(전환율 대비 쿠폰 비용)**
   * **적용**: 매출 방어 최우선 타겟은 `VIP`(매출 핵심 기여), 단기 전환 ROI 타겟은 `Churned`(최대 인원 37.1%, 평균 미구매 181.38일)로 우선순위를 분리.

8. **#19 머신러닝(ML) 확장 설계 (타겟 및 피처 목록)**
   * **예측 타겟(Target) 예시**:
     * `is_churned` (이진 분류: 향후 90일 내 재구매 여부 / 이탈 예측)
     * `next_purchase_amount` 또는 `LTV` (회귀: 고객 생애가치 예측)
   * **사용 피처(Features) 목록**:
     * **RFM 파생 피처**: `Recency`, `Frequency`, `Monetary`, `R_score`, `F_score`, `M_score`, `RFM_score`
     * **이미지 멀티모달 피처**: `img_mean`, `img_std`, `img_downsampled_mean`
     * **텍스트·메타 피처**: `word_count`, `name_length`, `category`, `gender`, `season`

---

## 3. 적용 체크리스트

- [x] `src/pipeline.py`에 `handle_missing = handle_missing_values` 별칭 추가
- [x] `src/visualize.py` 생성 (6종 차트 + 박스플롯 전후 비교 PNG 저장)
- [x] `README.md` 및 `REPORT.md`에 `#2`, `#5`, `#6`, `#13`, `#14`, `#17`, `#18`, `#19` 내용 반영
- [x] `python -m unittest discover -s tests -v` 및 `python scripts/verify_requirements.py` 재실행 후 커밋/푸시 준비 완료
