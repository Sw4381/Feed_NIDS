# FEED-NIDS Pipeline v3

## 개요

네트워크 침입 탐지 시스템의 오탐을 줄이기 위한 자동화 파이프라인입니다.

**핵심 동작:**
1. **Detection**: XGBoost 모델로 Attack/Normal 예측
2. **Prioritizer**: Attack 중 하위 5% 선정 (확신도 낮은 케이스)
3. **Knowledge Base**: Train 데이터 기반 라벨 자동 조정
4. **Feedback**: 이전 라운드 피드백 기반 라벨 자동 조정
5. **Merge**: 최종 결과 통합

**주요 특징:**
- Annoy 기반 고속 유사도 검색 (O(log N))
- 캐싱 시스템 (40분 → 5초)
- SHAP 기반 설명 가능성

---

## 설치
```bash
pip install pandas numpy scikit-learn joblib shap annoy xgboost
```

---

## 실행 방법

### 기본 실행
```bash
# 전체 파이프라인 (Detection → KB → Feedback)
python pipeline_orchestrator_v2.py --all

# 특정 라운드만
python pipeline_orchestrator_v2.py --rounds Round_1 Round_2
```

### 모드 선택
```bash
# KB만 적용
python pipeline_orchestrator_v2.py --all --mode kb-only

# Feedback만 적용
python pipeline_orchestrator_v2.py --all --mode feedback-only
```

### 최적화 옵션
```bash
# Annoy 최적화 + 캐싱 활성화 (권장)
python pipeline_orchestrator_v2.py --all --use-optimized-search

# 캐시 강제 재빌드
python pipeline_orchestrator_v2.py --all --force-rebuild-cache

# 캐싱 비활성화
python pipeline_orchestrator_v2.py --all --no-cache
```

---

## 주요 파라미터

### Detection
```bash
--skip-detection          # Detection 스킵
--force-detection         # Detection 재실행
--det-threshold 0.5       # Attack 임계값
```

### Prioritizer
```bash
--gate-bottom-percent 5.0 # 하위 5% 선정
--gate-no-shap            # SHAP 비활성화
```

### Knowledge Base
```bash
--kb-alpha 0.3            # IP 가중치
--kb-beta 0.4             # Cosine 가중치
--kb-gamma 0.3            # SHAP 가중치
--kb-threshold 0.6        # 유사도 임계값
```

### Feedback
```bash
--alpha 0.3               # IP 가중치
--beta 0.4                # Cosine 가중치
--gamma 0.3               # SHAP 가중치
--threshold 0.6           # 유사도 임계값
--auto-top-n 50           # AutoFeedback 선정 개수
```

### 최적화
```bash
--use-optimized-search    # Annoy 활성화
--annoy-n-trees 100       # Annoy 트리 개수
--annoy-top-k 1000        # Annoy 검색 Top-K
--cache-dir ./cache       # 캐시 디렉토리
```

---

## 출력 파일
```
round_predictions/
├── {Round}_with_predictions.csv          # Detection 결과
├── {Round}_kb_applied.csv                # KB 적용 결과
└── {Round}_with_predictions_applied.csv  # 최종 결과

feedback_cases/
└── {Round}_low_confidence_cases.csv      # Prioritizer 결과

round_predictions_applied/
└── {Round}_position_aware_optimal.csv    # Feedback 적용 결과

cache/
├── kb_index/                             # KB 캐시
└── feedback_corpus/                      # Feedback 캐시
```

---

## 유사도 계산
```
S = α × S_IP + β × S_Cosine + γ × S_SHAP

- S_IP: IP 주소 일치도 (0 or 1)
- S_Cosine: StandardScaler 정규화 후 코사인 유사도
- S_SHAP: SHAP Top-K 피처 랭킹 오버랩
```

---

## 문제 해결
```bash
# Annoy 미설치
pip install annoy

# 캐시 무효화
python pipeline_orchestrator_v2.py --all --force-rebuild-cache

# 메모리 부족
python pipeline_orchestrator_v2.py --all --chunk-size 500

# SHAP 에러
python pipeline_orchestrator_v2.py --all --gate-no-shap

# 라벨 조정 너무 적음
python pipeline_orchestrator_v2.py --all --kb-threshold 0.4 --threshold 0.4

# 라벨 조정 너무 많음
python pipeline_orchestrator_v2.py --all --kb-threshold 0.8 --threshold 0.8
```

---

## 디렉토리 구조
```
필수 디렉토리:
- test_rounds/          # 입력 데이터 (Round_*.csv)
- Train_Cases/          # KB 데이터 (labeled CSV)
- models/               # 학습된 모델 (.joblib)

자동 생성:
- round_predictions/
- feedback_cases/
- round_predictions_applied/
- cache/
```

---

## 요구사항

- Python 3.8+
- pandas, numpy, scikit-learn, joblib, shap, annoy, xgboost
