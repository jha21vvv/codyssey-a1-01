# 🎙️ NOTEBOOKLM_GUIDE.md — Audio Overview 팟캐스트 학습 가이드

> 본 문서는 `PROJECT_WORKFLOW_PROMPT_TEMPLATE.md`의 **Step 6**에 따라 Google NotebookLM의 **Audio Overview(딥다이브 팟캐스트)**를 통해 이동 중에도 프로젝트 핵심을 귀로 복습할 수 있도록 구성한 안내서입니다.

---

## ✅ NotebookLM 업로드 소스 파일 체크리스트 (필수 확인)

Google NotebookLM은 `.py` 확장자 직접 업로드를 지원하지 않으므로, 아래 **4개 파일만** 소스로 업로드하세요.

- [ ] `NOTEBOOKLM_FULL_SOURCE.txt` (2단 주석이 포함된 순수 프로덕션 소스 코드 전체 통합본)
- [ ] `스터디.md` (과제 요구사항 전문 + 1:1 매핑표 + 3개월 차 주니어 눈높이 일상 비유)
- [ ] `REPORT.md` (아키텍처 의사결정 + 엣지 케이스 방어 + 벤치마크 결과)
- [ ] `README.md` (데이터셋 출처/구성 + 비즈니스 인사이트 3가지 + 평가자 검증 가이드)

> 소스 코드 수정 후에는 `python scripts/build_notebooklm_source.py`를 실행해 `NOTEBOOKLM_FULL_SOURCE.txt`를 즉시 최신화할 수 있습니다.

---

## 1. 호스트 2인 배경 스토리 및 대화 유도 질문 리스트

* **호스트 A (시니어 AI/데이터 아키텍트)**: 제약 조건(OpenCV/Pillow/Scikit-learn 금지) 속에서 순수 NumPy·Pandas만으로 OOP 파이프라인을 설계한 멘토.
* **호스트 B (3개월 차 주니어 엔지니어)**: "왜 `for`문 대신 `axis=(1,2,3)`을 쓰는지", "결측치 대치에서 왜 전체 평균 대신 GroupBy를 쓰는지" 날카롭게 질문하는 학습자.

**대화 유도 질문 5가지:**
1. *"단일 공개 데이터셋에 이미지와 주문 로그가 같이 없을 때, 어떻게 카탈로그 이미지를 베이스로 멀티모달 데이터셋을 구성하고 README에 투명하게 밝혔나요?"*
2. *"OpenCV나 Pillow 없이도 NumPy 배열 연산과 슬라이싱만으로 이미지 평균·표준편차·다운샘플링을 처리할 수 있는 원리가 뭔가요?"*
3. *"IQR 이상치 처리 시 `action='remove'`를 선택했을 때 DataFrame과 이미지 배열의 인덱스가 어긋나는 버그는 어떻게 막았나요?"*
4. *"RFM 4분위수(`pd.qcut`)에서 동점자 때문에 경계가 겹치는 오류를 `rank(method='first')`로 어떻게 해결했나요?"*
5. *"VIP(32.5%)와 Churned(37.1%) 세그먼트 분석 결과에서, 왜 모든 비즈니스 인사이트에 '검증(반증 가능성/추가 데이터)' 문장이 꼭 들어가야 하나요?"*

---

## 2. 핵심 기술적 결정 3가지 & 일상 비유 모음

1. **NumPy 축 벡터 연산 (`images.mean(axis=(1,2,3))`)**: 사진 2,000장을 1장씩 돋보기로 세는 `for`문 대신, 2,000장을 특수 스캐너에 한 번에 올려 0.01초 만에 평균과 명암 편차를 뽑는 방식.
2. **GroupBy 결측치 대치**: 옷 가격표가 떨어졌을 때 매장 전체 평균이 아니라 "같은 코너(`category`)의 중간 가격"으로 채우고, 코너 전체가 비었을 때만 전체 대표값으로 fallback.
3. **Tie-Safe RFM 분위수화 (`rank(method="first")`)**: 동점자가 많아 4등분 경계가 겹치면 줄 세운 순서대로 번호표를 매겨 에러 없이 4개 그룹(`VIP`, `Loyal`, `New`, `Churned`)을 나누는 안전장치.

---

## 3. 3개월 차 주니어가 반드시 깨달아야 할 청취 포인트 3가지

1. **제약 조건이 곧 실력이다**: 도메인 전용 라이브러리(`cv2`, `sklearn`)를 못 쓰는 환경에서도 NumPy 브로드캐스팅·축 연산과 Pandas `groupby`/`qcut`만으로 완전한 엔드투엔드 파이프라인을 만들 수 있다.
2. **데이터 정합성(Alignment)이 생명이다**: 표 데이터(`DataFrame`)에서 이상치 행을 지울 때 짝지어진 이미지 텐서(`ndarray`)도 함께 슬라이싱하지 않으면 조용히 잘못된 데이터가 매칭된다.
3. **인사이트는 근거·실행·검증의 3박자다**: 수치만 나열하면 보고서가 아니라 메모일 뿐이며, 대상 세그먼트와 실행 액션, 그리고 가설을 반증할 추가 검증 데이터까지 제시해야 실무 의사결정에 쓰인다.

---

## 4. NotebookLM Customize Audio Overview 전용 프롬프트

### 영문 프롬프트 (English)
```text
Focus on the engineering decisions behind `DataAnalyzer` in `NOTEBOOKLM_FULL_SOURCE.txt` and `REPORT.md`.
Have a senior data architect explain to a 3-month junior developer how they built a multimodal e-commerce pipeline using ONLY NumPy and Pandas (no OpenCV, Pillow, or Scikit-learn).
Highlight: (1) NumPy axis vectorization vs Python loops, (2) GroupBy missing value imputation & IQR capping with image-array synchronization, (3) tie-safe RFM segmentation with `rank(method="first")`, and (4) why every business insight needs evidence, action, and falsifiability. Use the everyday analogies from `스터디.md`.
```

### 국문 프롬프트 (Korean)
```text
`NOTEBOOKLM_FULL_SOURCE.txt`와 `스터디.md`, `REPORT.md`를 바탕으로 시니어 데이터 엔지니어와 3개월 차 주니어 개발자가 대화하는 딥다이브 팟캐스트를 진행해 주세요.
OpenCV·Pillow·Scikit-learn 없이 순수 NumPy와 Pandas만으로 멀티모달(텍스트+이미지+거래) 파이프라인 `DataAnalyzer`를 설계한 이유를 중심으로 다뤄주세요.
특히 (1) for문 없는 NumPy 축 연산과 슬라이싱 다운샘플링, (2) 카테고리별 GroupBy 결측치 대치 및 IQR 이상치 처리 시 이미지 배열 동기화, (3) `rank(method='first')`를 활용한 동률 안전 RFM 4그룹 분류, (4) `스터디.md`의 일상 비유와 비즈니스 인사이트 3요소(근거·실행·검증)를 재미있고 명확하게 설명해 주세요.
```
