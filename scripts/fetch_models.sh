#!/bin/sh
# CubiCasa5K 코드 + 가중치(209MB) 내려받기. 실행 위치: ai-server/
set -e
mkdir -p third_party weights
[ -d third_party/CubiCasa5k ] || git clone --depth 1 https://github.com/CubiCasa/CubiCasa5k.git third_party/CubiCasa5k
[ -f weights/model_best_val_loss_var.pkl ] || gdown 1gRB7ez1e4H7a9Y09lLqRuna0luZO5VRK -O weights/model_best_val_loss_var.pkl
