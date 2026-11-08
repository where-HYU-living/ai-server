#!/bin/sh
# 모델 코드 + 가중치 내려받기. 실행 위치: ai-server/
#   CubiCasa5K 코드: GitHub 원본 / 가중치(209MB): Hugging Face (CC BY-NC 4.0)
#   적정 월세 모델: 같은 저장소의 rent/model.joblib (아직 없으면 건너뜀)
set -e
REPO=${HF_MODEL_REPO:-combe4259/chatatda-models}
mkdir -p third_party weights
[ -d third_party/CubiCasa5k ] || git clone --depth 1 https://github.com/CubiCasa/CubiCasa5k.git third_party/CubiCasa5k
[ -f weights/model_best_val_loss_var.pkl ] || hf download "$REPO" model_best_val_loss_var.pkl --local-dir weights
[ -f weights/rent/model.joblib ] || hf download "$REPO" rent/model.joblib --local-dir weights \
  || echo "적정 월세 모델이 저장소에 없어요 (학습 전) — 건너뜀"
