# TankChallenge RAG 의사결정 지원 시스템

TankChallenge 전투/발사 로그를 기반으로 현재 상황과 유사한 과거 명중/실패 사례를 검색하고, 발사 여부와 조준 보정값을 추천하는 RAG 기반 의사결정 지원 시스템입니다.

본 프로젝트는 단순 규칙 기반 제어가 아니라, 실제 전투 로그를 자연어 사례 문서로 변환하고 Hugging Face SentenceTransformer로 임베딩한 뒤 FAISS와 ChromaDB에서 유사 사례를 검색하는 구조를 갖습니다.

## 핵심 기능

- TankChallenge CSV 발사 로그 기반 전투 사례 데이터셋 구축
- 발사 상황을 자연어 문서 형태로 변환
- Hugging Face SentenceTransformer 기반 임베딩 생성
- FAISS 기반 고속 벡터 검색
- ChromaDB 기반 영속 VectorDB 저장소 구성
- 현재 상황과 유사한 Top-k 과거 사례 검색
- 유사 사례의 명중/실패 분포 기반 발사 여부 추천
- yaw/pitch 조준 보정값 추천
- 성공 사례 weighted average 기반 보정
- 실패 사례 오차 방향을 반영한 보정값 보수화
- 거리 bucket별 조준 보정 및 confidence threshold 적용
- moving/stationary 표적 유형별 분리 threshold 적용
- 검색 품질 평가 지표 제공
- PCA 기반 임베딩 분포 시각화
- PCA 이미지와 검색 예시가 포함된 제출용 리포트 자동 생성
- Flask 웹 대시보드 및 JSON API 제공
- 기존 TankChallenge 서버 상태 API와 연동 가능한 live-query bridge 제공

## 전체 구조

```text
shot_analysis/*.csv
shot_analysis/moving_target_logs/*.csv
        |
        v
CSV 로그 파싱
        |
        v
자연어 전투 사례 문서 생성
        |
        v
SentenceTransformer 임베딩 생성
        |
        +--------------------+
        |                    |
        v                    v
   FAISS Index          ChromaDB Store
        |                    |
        +---------+----------+
                  |
                  v
          현재 상황 Query 입력
                  |
                  v
      유사한 과거 명중/실패 사례 Top-k 검색
                  |
                  v
      발사/보류 판단 + yaw/pitch 보정 추천
```

## 사용 기술

- Python
- Flask
- Hugging Face `sentence-transformers`
- FAISS `faiss-cpu`
- ChromaDB
- scikit-learn PCA
- Matplotlib
- Plotly
- HTML/CSS

기본 임베딩 모델:

```text
sentence-transformers/all-MiniLM-L6-v2
```

지원 검색 방식:

```text
faiss   - Hugging Face 임베딩 + FAISS 벡터 검색
chroma  - Hugging Face 임베딩 + ChromaDB VectorDB 검색
hybrid  - 숫자 특징/텍스트 유사도 기반 baseline 검색
```

## 파일 구성

```text
rag_decision_support/
  app.py                         # Flask 웹 대시보드 및 API
  tank_rag.py                    # 로그 파싱, 임베딩, 검색, 추천 로직
  pca_visualize.py               # PCA 시각화 생성 스크립트
  case_index.jsonl               # 자연어 전투 사례 인덱스
  faiss_cases.index              # FAISS 벡터 인덱스
  faiss_cases_meta.json          # FAISS 메타데이터 및 원본 사례
  chroma_store/                  # ChromaDB 영속 벡터 저장소
  pca_embedding_points.csv       # PCA 2차원 좌표 데이터
  rag_report.md                  # 평가/분석 리포트
  templates/index.html           # 웹 대시보드 템플릿
  static/styles.css              # 웹 대시보드 스타일
  static/pca_embedding_map.png   # PCA 정적 이미지 fallback
```

## 설치

프로젝트 루트에서 실행합니다.

```powershell
python -m pip install -r requirements.txt
```

주요 필요 패키지:

```text
flask
sentence-transformers
faiss-cpu
chromadb
matplotlib
scikit-learn
```

## 인덱스 생성 순서

프로젝트 루트에서 실행합니다.

```powershell
python rag_decision_support\tank_rag.py build
python rag_decision_support\tank_rag.py build-embeddings
python rag_decision_support\tank_rag.py build-chroma
python rag_decision_support\pca_visualize.py
python rag_decision_support\tank_rag.py report --backend faiss
```

각 명령의 의미:

```text
build             CSV 로그를 읽어 case_index.jsonl 생성
build-embeddings  SentenceTransformer 임베딩 및 FAISS 인덱스 생성
build-chroma      ChromaDB 영속 벡터 저장소 생성
pca_visualize.py  임베딩을 PCA로 2차원 축소하고 CSV/PNG 생성
report            검색 성능 및 데이터셋 요약 리포트 생성
```

## CLI 질의 예시

FAISS 기반 검색:

```powershell
python rag_decision_support\tank_rag.py query --backend faiss --distance 85 --body-error 3 --turret-error 0.8 --pitch-error -0.1 --enemy-speed 0.4
```

ChromaDB 기반 검색:

```powershell
python rag_decision_support\tank_rag.py query --backend chroma --distance 85 --body-error 3 --turret-error 0.8 --pitch-error -0.1 --enemy-speed 0.4
```

Hybrid baseline 검색:

```powershell
python rag_decision_support\tank_rag.py query --backend hybrid --distance 85 --body-error 3 --turret-error 0.8 --pitch-error -0.1 --enemy-speed 0.4
```

출력 예시:

```json
{
  "backend": "faiss",
  "recommendation": {
    "fire": true,
    "confidence": 1.0,
    "summary": "Similar-case success rate is 100%. Recommendation: fire.",
    "yaw_correction_deg": 0.242,
    "pitch_correction_deg": 0.09
  }
}
```

## 웹 대시보드 실행

```powershell
cd rag_decision_support
python .\app.py
```

브라우저 접속:

```text
http://127.0.0.1:5056
```

대시보드 기능:

- 현재 전투 상황 수동 입력
- 검색 backend 선택: FAISS / ChromaDB / Hybrid baseline
- 발사 권장 또는 발사 보류 판단 표시
- yaw/pitch 조준 보정값 표시
- 검색된 Top-k 유사 사례 표시
- 검색 품질 평가 지표 표시
- FAISS와 hybrid baseline 검색 결과 비교
- Plotly 기반 인터랙티브 PCA 임베딩 시각화
- Plotly CDN이 불가능할 경우 정적 PCA PNG fallback 표시

## 검색 품질 평가 지표

각 질의마다 다음 지표를 제공합니다.

```text
Top-k success/failure 개수
Top-k success rate
검색된 사례의 평균 impact error
성공 사례 평균 거리
FAISS와 hybrid 검색 결과 overlap 개수
FAISS 요약 지표와 hybrid 요약 지표 비교
```

이를 통해 단순히 RAG를 붙인 것이 아니라, 검색 결과의 품질을 평가하고 비교할 수 있습니다.

## PCA 시각화

384차원 임베딩을 PCA로 2차원 축소합니다.

```powershell
python rag_decision_support\pca_visualize.py
```

생성 파일:

```text
rag_decision_support/pca_embedding_points.csv
rag_decision_support/static/pca_embedding_map.png
```

웹 대시보드는 `pca_embedding_points.csv`를 읽어 Plotly 산점도를 렌더링합니다.

점에 마우스를 올리면 다음 정보를 볼 수 있습니다.

```text
case_id
hit_label
target_type
distance
impact_error
source file
```

현재 PCA 결과:

```text
case_count: 116
embedding_dimension: 384
PC1 explained variance: 약 60.7%
PC2 explained variance: 약 17.6%
PC1 + PC2 total: 약 78.2%
```

## API

메트릭 조회:

```text
GET /api/metrics
```

질의:

```text
GET /api/query?backend=faiss&distance=85&body_error=3&turret_error=0.8&pitch_error=-0.1&enemy_speed=0.4
GET /api/query?backend=chroma&distance=85&body_error=3&turret_error=0.8&pitch_error=-0.1&enemy_speed=0.4
```

PCA point 조회:

```text
GET /api/pca-points
```

인덱스 재생성:

```text
POST /api/rebuild
```

실시간 서버 연동:

```text
GET /api/live-query?source=http://127.0.0.1:5000/fire_status&backend=faiss
```

`live-query`는 기존 TankChallenge Flask 서버의 상태 API에서 현재 거리, 조준 오차, 적 속도 등의 값을 읽어 RAG query로 변환합니다.

지원하는 대표 필드명:

```text
distance / distance_fire / target_distance
body_error / body_error_fire
turret_error / turret_error_fire
pitch_error / pitch_error_fire
enemy_speed / enemy_speed_fire
lead_distance / lead_distance_fire
```

## 현재 데이터셋

사용 로그:

```text
shot_analysis/shot_log_*.csv
shot_analysis/moving_target_logs/moving_shot_log_*.csv
```

현재 인덱싱된 사례 수:

```text
116 cases
```

## 추천 로직

1. 현재 상황을 자연어 query document로 변환합니다.
2. SentenceTransformer로 query embedding을 생성합니다.
3. FAISS 또는 ChromaDB에서 Top-k 유사 과거 사례를 검색합니다.
4. 검색된 사례의 성공/실패 분포를 계산합니다.
5. 성공 사례만 가중 평균하여 yaw/pitch 기준 보정값을 계산합니다.
6. 실패 사례의 오차 방향을 반영해 반복 실패 패턴으로 향하지 않도록 보정합니다.
7. 거리 bucket을 close/mid/far로 나누고, moving/stationary 표적 유형별 threshold를 다르게 적용합니다.
8. 유사 성공률과 현재 조준 오차 조건을 함께 고려해 발사/보류를 추천합니다.

## 자동 리포트 생성

다음 명령으로 제출용 분석 리포트를 자동 생성합니다.

```powershell
python rag_decision_support\tank_rag.py report --backend faiss
```

생성 파일:

```text
rag_decision_support/rag_report.md
```

리포트 포함 내용:

```text
데이터셋 요약
보수적 fire/hold 판단 일치율
평균 impact error
FAISS / ChromaDB / Hybrid 검색 품질 비교표
검색 예시와 Top-k 유사 사례
추천 결과 JSON 예시
PCA 이미지 링크
추천 로직 설명
```

## 포트폴리오 요약 문장

> TankChallenge 전투 로그를 자연어 사례 문서로 변환하고 Hugging Face SentenceTransformer로 임베딩을 생성한 뒤, FAISS와 ChromaDB 기반 Vector Search를 통해 현재 상황과 유사한 과거 명중/실패 사례를 검색하는 RAG 기반 의사결정 지원 시스템을 구현했습니다. 검색된 유사 사례를 바탕으로 발사 여부와 yaw/pitch 조준 보정값을 추천하며, 검색 품질 평가 지표와 PCA 기반 임베딩 시각화를 함께 제공했습니다.

## 참고 사항

- 첫 실행 시 Hugging Face 모델 다운로드 때문에 시간이 걸릴 수 있습니다.
- Hugging Face 토큰 없이도 실행되지만, 미인증 요청 경고가 출력될 수 있습니다.
- Plotly는 CDN으로 로드됩니다. CDN 접근이 불가능하면 정적 PCA PNG가 fallback으로 표시됩니다.
- FAISS는 로컬 고속 검색 backend이고, ChromaDB는 영속 VectorDB workflow를 보여주기 위한 backend입니다.
- Windows PowerShell에서 한글이 깨져 보이면 다음 명령으로 확인하세요.

```powershell
Get-Content .\README.md -Encoding utf8
```
