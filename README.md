# ai-server — 평면도 AI 서버 (FastAPI)

평면도 이미지(루트 A) 또는 면적·방 구성(루트 B)을 받아 프론트 `FloorPlan` JSON 을 만든다.
가구(fixtures)는 만들지 않는다 (프론트가 방 이름·면적으로 예시 배치).

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
| `GET /health` | 가중치 유무 |

모델이 없으면(`weights/` 없음) 루트 A 는 503, 루트 B 는 정상 동작한다.

## 실행
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
sh scripts/fetch_models.sh            # CubiCasa5K 코드+가중치(209MB)
.venv/bin/uvicorn app.main:app --port 8000
.venv/bin/python -m pytest -q tests
.venv/bin/python scripts/evaluate.py  # 27장 vs 정답 JSON (숫자 비교)
```
## 배포 (Railway)
`Dockerfile` + `railway.json`.
백엔드에는 환경변수 `AI_SERVER_URL=https://<서비스 주소>` 를 둔다.

## 라이선스 주의
CubiCasa5K 코드·가중치는 CC BY-NC 4.0 (비상업). 상업 배포 전에는 교체해야 한다.
