# FEED-NIDS 파이프라인 수정 사항

## 주요 변경 내용

### 1. Label 재조정 로직 추가 ✅

**파일**: `orchestrator/policy.py`

- **기능**: 유사 사례의 Label을 참조하여 자동으로 Alert의 Label 재조정
- **재조정 기준**:
  - `label_confidence_threshold` (기본 0.6 = 60%) 이상의 유사도를 가진 사례 발견 시
  - 가중 투표 방식으로 최종 Label 결정
  - Normal로 재조정 또는 Attack 확인

**Decision 타입**:
- `AUTO_ADJUST_TO_NORMAL`: 고신뢰도로 Normal로 재조정
- `AUTO_CONFIRM_ATTACK`: 고신뢰도로 Attack 확인
- `REFER_TO_SIMILAR`: 중신뢰도, 참고만 (원래 Label 유지)
- `LOW_SIMILARITY`: 저신뢰도, 원래 Label 유지
- `NO_SIMILAR_CASE`: 유사 사례 없음
- `NO_LABELED_CASE`: 유사 사례는 있으나 Label 없음

### 2. Evidence에 Label 포함 ✅

**파일**: `tools/evidence_writer.py`

- **변경 전**: Label이 빈 문자열("")로 저장됨
- **변경 후**: `adjusted_label`을 Label로 저장
- **효과**: 다음 라운드에서 유사 사례 검색 시 해당 Label 참조 가능

### 3. 샘플링 비율 3%로 변경 ✅

**파일**: `tools/evidence_writer.py`, `configs/config.yaml`

- **변경 전**: `insertion_rate_percent = 10.0` (10%)
- **변경 후**: `insertion_rate_percent = 3.0` (3%)
- **샘플링 방식**: confidence 기준 오름차순 정렬 후 하위 3% 선택

### 4. 파이프라인 통합 ✅

**파일**: `run.py`

- Decision 결과를 Evidence Writer로 전달
- adjusted_label 반영된 CSV 생성
- Label 변경 통계 출력
- 라운드 로그에 `similar_found`, `label_adjusted` 추가

---

## 프로세스 흐름 (수정 후)

```
1. 탐지기 모델 예측
   ↓
2. 게이팅 (Attack 중 하위 5% 선정)
   ↓
3. Evidence Pack 생성
   ↓
4. RAG 유사사례 검색
   ↓
5. ★ Label 재조정 (유사사례의 Label 참조)
   ↓
6. 유사사례 없는 케이스 중 하위 3% 선정
   ↓
7. ★ Evidence 저장 (adjusted_label 포함)
   ↓
8. RAG 인덱스 재구축 (다음 라운드에서 활용)
```

---

## 주요 설정값

### `configs/config.yaml`

```yaml
decision:
  label_confidence_threshold: 0.6  # Label 재조정 최소 유사도 (60%)
  top_refs: 3                      # 참조할 상위 유사사례 개수

evidence:
  not_found_sim_threshold: 0.15    # 유사사례 없음 판단 임계값 (15%)
  insertion_rate_percent: 3.0      # Evidence 저장 비율 (3%)
```

---

## 출력 파일

### 라운드별 산출물 (`round_outputs/<Round>/`)

1. **targets.csv**: 게이팅된 분석 대상
2. **evidence_packs.jsonl**: Evidence Pack 목록
3. **rag_candidates.jsonl**: RAG 검색 결과 (상위 20개)
4. **decisions.jsonl**: ★ Label 재조정 결과
   - `adjusted_label`: 재조정된 Label
   - `confidence`: 재조정 확신도
   - `reference_cases`: 참조한 유사사례
   - `reasoning`: 재조정 근거
5. **predictions_with_adjusted.csv**: ★ adjusted_label 포함된 전체 예측 결과
6. **evidence_inserted_ids.txt**: Evidence DB에 저장된 ID 목록
7. **round_summary.json**: 라운드 요약 통계

### Evidence DB (`evidence_db/`)

1. **packs.jsonl**: ★ Label이 포함된 Evidence Pack 저장소
2. **evidence_index.jsonl**: 인덱스 파일

### 전체 로그 (`round_outputs/`)

- **round_log.csv**: 라운드별 집계
  - `gated`: 게이팅 대상 수
  - `similar_found`: 유사사례 발견 수
  - `label_adjusted`: Label 재조정 수
  - `not_found`: 유사사례 없음 수
  - `inserted`: Evidence DB 저장 수

---

## 사용 방법

### 1. 실행

```bash
python run.py
```

### 2. 설정 변경

`configs/config.yaml` 파일에서 파라미터 조정

### 3. Label 재조정 확인

각 라운드의 `decisions.jsonl` 파일 확인:

```json
{
  "alert_id": "Round_1-0001",
  "decision": "AUTO_ADJUST_TO_NORMAL",
  "adjusted_label": "Normal",
  "original_label": "Attack",
  "confidence": 0.85,
  "reference_cases": [
    {
      "ref_id": "...",
      "label": "Normal",
      "similarity": 0.87
    }
  ],
  "reasoning": "유사도 0.87로 2개 사례에서 Normal로 재조정"
}
```

---

## 검증 포인트

### ✅ 확인 사항

1. **Evidence DB의 Label 확인**
   ```bash
   cat evidence_db/packs.jsonl | jq '.label'
   ```

2. **Label 재조정 통계 확인**
   ```bash
   cat round_outputs/round_log.csv
   ```

3. **재조정 케이스 상세 확인**
   - `round_outputs/<Round>/decisions.jsonl` 에서 `adjusted_label != original_label` 케이스 확인

---

## 문제 해결

### Label이 여전히 비어있는 경우

1. `decision.label_confidence_threshold` 값 조정 (낮추기)
2. RAG 검색이 제대로 되는지 확인 (`rag_candidates.jsonl`)
3. Evidence DB에 Label이 있는 사례가 충분한지 확인

### 재조정이 너무 적은 경우

- `decision.label_confidence_threshold`를 0.5 이하로 낮추기
- `rag.coarse_k` 증가 (더 많은 후보 검색)

### 재조정이 너무 많은 경우

- `decision.label_confidence_threshold`를 0.7 이상으로 높이기
- `evidence.not_found_sim_threshold` 증가 (더 엄격한 필터링)

---

## 변경 파일 목록

1. ✅ `orchestrator/policy.py` - 신규 생성 (Label 재조정 로직)
2. ✅ `tools/evidence_writer.py` - 수정 (Label 포함 저장)
3. ✅ `run.py` - 수정 (Decision-Evidence 연계)
4. ✅ `configs/config.yaml` - 수정 (3% 샘플링, 임계값 설정)
5. ✅ `tools/detection.py` - 유지
6. ✅ `tools/prioritizer.py` - 유지
7. ✅ `tools/evidence_pack.py` - 유지
8. ✅ `tools/feedback_search.py` - 유지
