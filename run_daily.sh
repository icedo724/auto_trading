#!/usr/bin/env bash
# 매일 장 마감 후(15:40) 신호를 산출해 로그와 CSV로 남긴다.
#   crontab:  40 15 * * 1-5 /경로/auto_trading/run_daily.sh
#
# 이 스크립트는 주문을 내지 않는다. 신호만 만든다.
# 주문 자동화는 모의투자 검증을 마친 뒤에 붙일 것 (docs/LIVE_TRADING.md 3단계).
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${QUANT_CONFIG:-configs/experiment_kr.yaml}"
BEST="${QUANT_BEST:-reports/kr/optimization_best.json}"

# 킬 스위치: STOP 파일이 있으면 아무것도 하지 않는다
if [[ -f STOP ]]; then
  echo "[$(date '+%F %T')] STOP 파일 존재 — 실행 중단"
  exit 0
fi

[[ -d .venv ]] && source .venv/bin/activate

mkdir -p logs reports/live
LOG="logs/$(date +%Y-%m-%d).log"

{
  echo "================ $(date '+%F %T') ================"
  if [[ ! -f "$BEST" ]]; then
    echo "최적 파라미터 파일이 없습니다: $BEST"
    echo "먼저 실행하세요:  python -m quant optimize -c $CONFIG"
    exit 1
  fi
  python -m quant signal -c "$CONFIG" --best-file "$BEST" --refresh --out reports/live
  echo "완료: $(date '+%F %T')"
} >> "$LOG" 2>&1

echo "신호 산출 완료 → $LOG, reports/live/signals.csv"
