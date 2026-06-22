# VLM Classification And Projection Validation Design

## 목적

기존 OCR 기반 SFT/분류 방식만으로는 체크박스, 서식 배치, 직인, 표 구조처럼 이미지에서만 보이는 단서가 충분히 반영되지 않는다. 따라서 문서 이미지를 직접 입력받는 VLM을 사용해 4축 분류를 수행하고, 기존 projection 모델과 dictionary 구조는 검증 및 정규화 레이어로 유지한다.

대상 4축은 다음과 같다.

- `subject`
- `document_type`
- `business_domain`
- `modifier`

## 핵심 방향

기존 `/classification/dictionary` 폴더는 변경하지 않는 것을 기본 원칙으로 한다.

현재 dictionary는 projection 검증뿐 아니라 key graph 및 visualization에도 사용된다. 따라서 VLM 학습 데이터나 VLM evidence를 dictionary에 직접 섞으면 visualization, fallback 분류, projection 입력 구조가 깨질 수 있다.

기존 dictionary는 다음 역할로 유지한다.

- key canonical registry
- signal dictionary
- projection validation 기준
- key embedding graph 생성 기준
- frontend visualization 데이터 소스

VLM 관련 결과물은 별도 데이터로 분리한다.

```text
aihub_88/
  category/
    images/
    new_labels/
    vlm_labels/
    vlm_sft/
```

## 전체 파이프라인

```text
image
  -> VLM
  -> 4-axis key + evidence.description
  -> LLM signalizer
  -> key + signals 후보
  -> projection validation
  -> dictionary normalization
  -> final result + review flags
```

각 컴포넌트의 역할은 명확히 분리한다.

| 단계 | 역할 |
| --- | --- |
| VLM | 이미지를 보고 4축 key와 시각적/문맥적 evidence 생성 |
| LLM signalizer | VLM evidence와 OCR text에서 projection용 signal 후보 추출 |
| Projection | key와 signals 조합이 dictionary 기준으로 타당한지 검증 |
| Dictionary | key, signals, graph, visualization의 안정 기준 |

## VLM 출력 스키마

VLM은 이미지를 보고 각 축에 대해 key와 evidence를 출력한다.

```json
{
  "subject": {
    "key": "과원폐업지원",
    "evidence": {
      "description": "문서 제목과 본문에 FTA 과원폐업지원사업 보조금 지급 관련 내용이 보입니다."
    }
  },
  "document_type": {
    "key": "공문",
    "evidence": {
      "description": "수신자, 제목, 시행, 직인 등 공문 서식 요소가 보입니다."
    }
  },
  "business_domain": {
    "key": "농림축산",
    "evidence": {
      "description": "과수원 폐업 지원 대상자와 보조금 지급 내용이 포함되어 있습니다."
    }
  },
  "modifier": {
    "key": "보조금",
    "evidence": {
      "description": "보조금 지급 요청과 지급액 관련 항목이 확인됩니다."
    }
  }
}
```

VLM은 이미지 기반 단서를 사용할 수 있다. 다만 evidence는 이후 projection 검증에 바로 넣지 않고, LLM signalizer를 통해 signals 후보로 변환한다.

## LLM Signalizer

`evidence.description`을 기계적 규칙으로 signals로 변환하면 품질 문제가 생길 수 있다. 행정문서는 표현이 다양하고, 체크박스나 서식 기반 근거가 OCR에 잡히지 않는 경우도 많기 때문이다.

따라서 별도 LLM signalizer를 둔다.

역할은 분류가 아니라 signal 추출이다.

```text
입력:
  - axis
  - VLM predicted key
  - VLM evidence.description
  - OCR rec_texts from new_labels

출력:
  - projection 검증에 사용할 signals 후보
  - visual_signals 후보
  - confidence
  - reason
```

출력 예시는 다음과 같다.

```json
{
  "axis": "business_domain",
  "key": "농림축산",
  "signals": [
    "FTA 과원폐업지원사업",
    "보조금 지급",
    "지원 대상자"
  ],
  "visual_signals": [
    "보조금 지급 관련 항목이 체크됨"
  ],
  "confidence": "high",
  "reason": "분류 key를 뒷받침하는 명시적 문구가 문서 제목과 본문 OCR에 존재합니다."
}
```

### Signalizer 정책

- 새로운 사실을 만들지 않는다.
- 가능하면 `new_labels`의 OCR `rec_text`에 존재하는 표현을 우선 사용한다.
- OCR에는 없지만 이미지에서 확인되는 근거는 `visual_signals`로 분리한다.
- 주어진 key를 뒷받침할 evidence가 약하면 `signals`를 빈 배열로 둔다.
- signalizer는 key를 재분류하지 않는다.
- VLM key와 evidence가 충돌하면 `confidence`를 낮추고 review 대상으로 넘긴다.

권장 프롬프트 방향:

```text
너는 분류기가 아니라 signal 추출기다.
주어진 axis/key를 정당화할 수 있는 짧은 근거 문구만 추출하라.
가능하면 OCR 텍스트에 존재하는 표현을 그대로 사용하라.
OCR에는 없지만 이미지 evidence에만 있는 단서는 visual_signals에 넣어라.
근거가 없으면 signals를 빈 배열로 반환하라.
```

## Projection 입력 변환

기존 projection 모델이 `key + signals`를 기준으로 검증한다면, LLM signalizer 출력은 다음처럼 adapter를 통해 기존 입력 형태로 변환한다.

```json
{
  "business_domain_key": "농림축산",
  "signals": [
    "FTA 과원폐업지원사업",
    "보조금 지급",
    "지원 대상자"
  ]
}
```

`visual_signals`는 projection에 바로 넣지 않고, 별도 검토 필드 또는 향후 visual projection 모델의 입력으로 둔다.

## 검증 결과 상태

projection 결과는 단순 true/false보다 상태값으로 관리하는 것이 좋다.

| 상태 | 의미 |
| --- | --- |
| `accept` | VLM key와 signals가 dictionary/projection 기준으로 타당함 |
| `review` | 근거는 있으나 확신이 낮거나 dictionary와 거리가 있음 |
| `conflict` | VLM key와 signals가 서로 다른 축/key를 가리킴 |
| `unknown_key` | VLM key가 dictionary에 없음 |
| `weak_evidence` | key는 있으나 signals가 비어 있거나 근거가 약함 |

예시:

```text
VLM key: 농림축산
signals: ["과원폐업지원사업", "보조금 지급"]
projection: 농림축산과 가까움
=> accept
```

```text
VLM key: 농림축산
signals: ["도로점용", "도시계획시설"]
projection: 도시개발과 가까움
=> conflict
```

```text
VLM key: 주민복지
signals: []
visual_signals: ["복지 급여 항목 체크됨"]
=> weak_evidence 또는 review
```

## Dictionary 관리 원칙

`dictionary/*.json`에는 검수된 key와 signals만 반영한다.

VLM이나 LLM signalizer가 생성한 새 key/signal은 즉시 dictionary에 넣지 않는다. 먼저 candidate/pending 데이터로 모으고, 검수 후 승격한다.

권장 흐름:

```text
vlm result
  -> signalizer output
  -> projection result
  -> pending dictionary candidates
  -> human review
  -> dictionary update
  -> key_embedding_graph rebuild
```

후보 저장 예시:

```json
{
  "axis": "modifier",
  "candidate_key": "체크박스_선택",
  "candidate_signals": [
    "해당 항목 체크됨"
  ],
  "source_file": "5350034-2004-0001-0001.jpg",
  "status": "pending"
}
```

## 학습 데이터 분리

VLM 학습 데이터와 projection 학습 데이터는 목적이 다르다. 다만 하나의 master label에서 둘 다 파생할 수 있다.

### Master Label

```json
{
  "image": "images/5350034-2004-0001-0001.jpg",
  "ocr_label": "new_labels/5350034-2004-0001-0001.json",
  "target": {
    "subject": {
      "key": "과원폐업지원",
      "evidence": {
        "description": "문서 제목에 FTA 과원폐업지원사업 보조금 지급 내용이 보입니다.",
        "phrases": [
          "FTA 과원폐업지원사업",
          "보조금 지급"
        ],
        "visual_signals": []
      }
    }
  }
}
```

### VLM 학습 데이터

목적:

```text
image -> 4-axis key + evidence.description
```

VLM은 OCR에 없는 체크박스, 서식, 직인, 표 구조 등을 함께 학습한다.

### Projection 학습 데이터

목적:

```text
key + signals -> valid/invalid/conflict
```

projection은 VLM 결과가 dictionary 기준으로 타당한지 검증하는 모델로 유지한다.

## 기존 Visualization과의 충돌 방지

현재 visualization은 dictionary와 key embedding graph에 의존한다. 따라서 다음을 지켜야 한다.

- `subject_key`, `document_type_key`, `business_domain_key`, `modifier_key` 필드명 유지
- `signals` 배열 구조 유지
- VLM evidence를 dictionary에 직접 추가하지 않음
- 검수되지 않은 candidate key를 graph에 넣지 않음
- dictionary 수정 후에는 `key_embedding_graph.json` 재생성

이렇게 하면 VLM 파이프라인을 추가해도 기존 visualization 구조는 안정적으로 유지할 수 있다.

## 추천 구현 순서

1. VLM 출력 schema 확정
2. `vlm_labels` 저장 포맷 확정
3. LLM signalizer 구현
4. signalizer 출력을 projection 입력으로 바꾸는 adapter 구현
5. projection 결과 상태값 정의
6. candidate/pending dictionary 저장소 추가
7. 검수된 candidate만 dictionary에 반영
8. dictionary 변경 시 key embedding graph 재생성

## 결론

VLM은 4축 후보를 생성하고, LLM signalizer는 evidence를 projection용 signals로 압축한다. 기존 projection 모델은 검증 레이어로 유지하고, dictionary는 visualization과 key normalization의 안정 기준으로 둔다.

이 구조를 사용하면 이미지 기반 분류 품질을 높이면서도 기존 projection, fallback, visualization 기능이 깨지는 위험을 줄일 수 있다.
