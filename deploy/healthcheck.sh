#!/usr/bin/env bash
# 페이퍼 트레이딩이 살아있는지 점검. cron 으로 하루 한 번 돌리면 좋다.
#   0 9 * * * /home/me/auto_trading/deploy/healthcheck.sh
#
# 마지막 사이클이 MAX_AGE_MIN 분보다 오래됐으면 비정상으로 판단한다.
set -euo pipefail
cd "$(dirname "$0")/.."

STATE="${1:-state/coin3m.jsonl}"
MAX_AGE_MIN="${MAX_AGE_MIN:-180}"

if [[ ! -f "$STATE" ]]; then
  echo "CRITICAL: 저널 파일이 없습니다 ($STATE). 한 번도 실행되지 않았습니까?"
  exit 2
fi

now=$(date +%s)
mtime=$(stat -c %Y "$STATE" 2>/dev/null || stat -f %m "$STATE")
age_min=$(( (now - mtime) / 60 ))

last=$(tail -n 1 "$STATE")
echo "마지막 갱신: ${age_min}분 전"
echo "마지막 이벤트: $last"

if (( age_min > MAX_AGE_MIN )); then
  echo "CRITICAL: ${MAX_AGE_MIN}분 넘게 갱신이 없습니다. 프로세스가 죽었을 수 있습니다."
  echo "  확인: systemctl status quant-paper ; journalctl -u quant-paper -n 50"
  exit 2
fi

errors=$(grep -c '"event": "cycle_error"' "$STATE" 2>/dev/null || true)
if [[ "${errors:-0}" -gt 0 ]]; then
  echo "WARNING: 누적 사이클 오류 ${errors}건"
  exit 1
fi

echo "OK"
