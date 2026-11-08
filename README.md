# ai-server — 평면도·적정 월세 AI 서버 (FastAPI)

평면도 이미지(루트 A) 또는 면적·방 구성(루트 B)을 받아 프론트 `FloorPlan` JSON 을 만든다.
가구(fixtures)는 만들지 않는다 (프론트가 방 이름·면적으로 예시 배치).

적정 월세는 국토부 실거래(동·면적·보증금·월세·계약월·층·준공연도·건물유형·신규/갱신)에
동 단위 외부 피처(사자상 거리·가장 가까운 지하철역 거리·500m 안 버스 정류장 수)를 붙여
㎡당 환산 월세의 25·50·75% 분위를 GBM 분위 회귀로 추정한다 (`app/rent/`).

```
루트 A  이미지 ─ CubiCasa5K 분할 ─ 방 폴리곤·벽 벡터화 ─ 문·창 배치 ─ 문 열림(기하) ┐
                 PaddleOCR(한국어) ─ 방 이름 ────────────────────────────────────────┤
루트 B  면적·방 수 ─ 템플릿(templates/*.json) 선택·스케일 ──────────────────────────┴─ 보정·검수 ─ JSON
```

| 엔드포인트 | 설명 |
|---|---|
| `POST /v1/floorplan/estimate` | multipart: `image`, `id`, `name`, `areaM2`(선택) |
| `POST /v1/floorplan/estimate-from-meta` | `{id, name, areaM2, rooms?}` |
| `POST /v1/floorplan/validate` | FloorPlan → 문제 목록 |
| `POST /v1/rent/estimate` | `{dong, areaM2, contractYm, floor?, builtYear?, buildingType?, contractType?}` → 적정 월세·범위(25~75%) |
| `GET /health` | 추론 스위치·가중치·적정 월세 모델 유무 |

모델이 없으면(`weights/` 없음) 루트 A 는 503, 루트 B 는 정상 동작한다.

## 실행
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sh scripts/fetch_models.sh            # CubiCasa5K 코드(GitHub) + 가중치(209MB, Hugging Face)
.venv/bin/uvicorn app.main:app --port 8000
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/evaluate.py  # 27장 vs 정답 JSON (숫자 비교)
```

## 적정 월세 모델 학습
```bash
# 실거래 CSV (백엔드 로컬 DB, 비어 있으면 백엔드에서 INGEST_ENABLED=true 로 한 번 수집)
docker exec chatatda-db psql -U chatatda -d chatatda -c "copy (select dong, area_m2, deposit, rent, contract_ym, floor, built_year, building_type, contract_type from real_deals) to stdout csv header" > data/raw/deals.csv
KAKAO_REST_KEY=... .venv/bin/python scripts/build_dong_features.py data/raw/deals.csv   # data/dong_features.json
.venv/bin/python scripts/train_rent.py data/raw/deals.csv      # weights/rent/ (모델·검증 결과)
hf upload combe4259/chatatda-models weights/rent rent          # 서버는 FETCH_MODELS=true 일 때 받아 감
```
현재 모델 (2025-11~2026-10 실거래 19,968건): 마지막 3개월 검증에서 절대오차율 중앙값 11.7% (단순 동 중앙값 20.1%),
평균 절대오차 11.6만 (중앙값 19.3만). 자세한 값은 [모델 카드](https://huggingface.co/combe4259/chatatda-models).
scikit-learn 버전은 학습한 버전(1.7.2)으로 고정한다 (`requirements.txt`).
## 배포 (Railway)
`Dockerfile` + `railway.json`.
모델 파일은 [Hugging Face `combe4259/chatatda-models`](https://huggingface.co/combe4259/chatatda-models)에 두고,
Railway 변수 `FETCH_MODELS=true` 일 때만 빌드 중에 받는다 (기본 `false` — 받지 않음).
추론을 쓰려면 `FETCH_MODELS=true` 와 `INFERENCE=on` 을 함께 켠다.
백엔드에는 환경변수 `AI_SERVER_URL=https://<서비스 주소>` 를 둔다.

## 라이선스 주의
CubiCasa5K 코드·가중치는 CC BY-NC 4.0 (비상업). 상업 배포 전에는 교체해야 한다.
