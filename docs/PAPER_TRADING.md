# 3개월 페이퍼 트레이딩 — 개인 서버 운용 가이드

실데이터 · **가상 자금**으로 전략을 24시간 돌린다. 실제 주문은 어디에서도 나가지 않으며,
증권사 API 키조차 필요 없다.

> **왜 이 단계가 필요한가.** 백테스트는 과거를 가장 유리하게 재구성한 결과다.
> 페이퍼 트레이딩은 **미래를 향해** 돌리므로 커브피팅이 원리적으로 불가능하다.
> 3개월 뒤 답해야 할 질문은 "얼마 벌었나"가 아니라
> **"백테스트가 약속한 것과 비슷하게 나왔나"** 하나다.

## 왜 코인인가

3개월 무인 운용에는 코인이 압도적으로 유리하다.

| | 코인 (업비트) | 국내주식 (KIS) |
|---|---|---|
| 시작까지 | **지금 바로** | 계좌 개설 + API 승인 (수일) |
| 인증 | **불필요** (공개 시세 API) | APP KEY/SECRET 발급 필요 |
| 운영 시간 | **24시간 365일** | 평일 6.5시간 |
| 3개월간 봉 수 | 약 90봉 | 약 60봉 |
| 소액 분산 | 소수점 매매 | 1주 단위 |

주식으로 하고 싶다면 `configs/experiment_kr.yaml` 로 소스만 바꾸면 된다.
페이퍼 트레이딩 자체는 브로커와 무관하다.

---

## 0. 시작 전 — 전략부터 정한다

페이퍼 트레이딩은 **이미 검증한 전략**을 굴리는 단계다. 아무 전략이나 넣고 3개월을
버리지 말 것.

```bash
python -m quant check-data --source upbit --symbol KRW-BTC   # 시세 확인
python -m quant fetch    -c configs/experiment_coin.yaml
python -m quant optimize -c configs/experiment_coin.yaml
python -m quant validate -c configs/experiment_coin.yaml     # WFE 확인
```

**WFE가 0.5 미만이면 페이퍼 트레이딩도 하지 말 것.** 3개월을 써서 "역시 안 된다"를
확인하게 된다. 다만 이미 안 될 걸 아는 전략을 굳이 돌려보고 싶다면, 그건 그것대로
**시스템 검증**으로서 가치가 있다. 그때는 목적을 "수익 확인"이 아니라
"운영 안정성 확인"으로 명확히 하고 시작하라.

---

## 1. 서버 준비

### 사양
페이퍼 트레이딩은 가볍다. **라즈베리파이나 최소 사양 VPS로 충분하다.**

| 항목 | 최소 | 비고 |
|---|---|---|
| CPU | 1코어 | 하루 몇 초만 일한다 |
| RAM | 512MB | 그리드 탐색을 안 돌리면 충분 |
| 디스크 | 2GB | 3개월 저널 + 시세 캐시 |
| 네트워크 | 아웃바운드 HTTPS | **인바운드 포트 열 필요 없음** |

> 인바운드를 열지 않는 것이 핵심이다. 이 시스템은 아무것도 서빙하지 않는다.
> 방화벽에서 들어오는 포트를 전부 막아도 정상 동작한다.

### 설치

```bash
git clone https://github.com/icedo724/auto_trading.git
cd auto_trading
git checkout claude/repository-name-project-belee9

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest tests/ -q          # 103 passed
python -m quant check-data --source upbit --symbol KRW-BTC
```

### 시간대 — **반드시 확인**

```bash
timedatectl                          # 또는 date
sudo timedatectl set-timezone Asia/Seoul
```

VPS 기본값은 대개 UTC다. 코인은 24시간이라 큰 문제는 아니지만, 저널 시각을 읽을 때
헷갈린다. 주식으로 한다면 **시간대가 틀리면 장 시간 판정이 통째로 어긋난다.**

---

## 2. 실행

### 한 사이클만 (동작 확인)

```bash
python -m quant paper \
    -c configs/experiment_coin.yaml \
    --best-file reports/coin/optimization_best.json \
    --name coin3m
```

```
[페이퍼] grid(band=0.2, levels=4, ma=20, trend_filter=0)
[페이퍼] 상태 state/coin3m.json · 저널 state/coin3m.jsonl
[페이퍼] 초기자본 100,000원 · 적립 100,000원/M  ※ 가상 자금 — 실제 주문 없음

  처리 5종목 · 건너뜀 0 · 체결 3건
    BUY  KRW-BTC    0.00023145 @ 86,420,000.00  수수료 10원
    BUY  KRW-ETH    0.00512300 @ 4,120,500.00   수수료 10원
  평가금액 99,970원
```

**같은 명령을 다시 실행해도 중복 체결되지 않는다.** 종목별 마지막 처리 봉을 기록하기
때문이다. cron 이 겹쳐 돌거나 서버가 재시작돼도 안전하다.

### 상주 실행 (3개월 무인)

```bash
python -m quant paper -c ... --best-file ... --name coin3m --loop --interval 3600
```

`--interval 3600` 은 1시간마다 깨어나 **새 봉이 마감됐는지만 확인**한다. 일봉 전략이라
하루 한 번만 실제로 일한다. 자주 깨우는 것은 낭비도 위험도 아니다(멱등하므로).

---

## 3. systemd 등록 — 3개월을 버티는 방법

3개월 = 서버 재부팅, 네트워크 끊김, 프로세스 크래시가 **반드시** 일어나는 기간이다.
`nohup ... &` 로는 못 버틴다.

### 방식 A: 상주 서비스 (권장)

```bash
sudo cp deploy/quant-paper.service /etc/systemd/system/quant-paper@$USER.service
sudo systemctl daemon-reload
sudo systemctl enable --now quant-paper@$USER
```

핵심 설정:

| 설정 | 효과 |
|---|---|
| `Restart=always` | 크래시해도 60초 뒤 되살아난다 |
| `enable` | **서버 재부팅 후 자동 시작** |
| `After=network-online.target` | 네트워크 준비 전에 시작하지 않는다 |
| `ProtectSystem=strict` | 지정한 디렉터리 외에는 쓰기 불가 |

### 방식 B: 타이머 (상주 프로세스가 싫다면)

```bash
sudo cp deploy/quant-paper-once.service /etc/systemd/system/quant-paper-once@$USER.service
sudo cp deploy/quant-paper.timer /etc/systemd/system/
sudo systemctl enable --now quant-paper.timer
```

`Persistent=true` 라서 **서버가 꺼져 있던 동안 놓친 실행을 부팅 후 보충**한다.
cron 에는 없는 기능이라 간헐적으로 켜는 서버에 유리하다.

### 확인

```bash
systemctl status quant-paper@$USER
journalctl -u quant-paper@$USER -f        # 실시간 로그
journalctl -u quant-paper@$USER --since today
```

---

## 3.5. 가동률 — 컴퓨터를 쭉 켜둬야 하나?

**아니다.** 일봉 전략이므로 하루에 **1분** 정도만 켜져 있으면 된다.

### 언제 켜져 있어야 하는가

업비트 원화마켓 일봉은 **매일 09:00 KST**(= 00:00 UTC)에 마감된다.
새 봉이 마감된 뒤 다음 마감 전까지, 즉 **하루 중 아무 때나 한 번** 실행되면 된다.

```
09:00 KST  일봉 마감 → 이때부터 새 신호 존재
   ↓  이 24시간 안에 한 번만 실행하면 됨
09:00 KST  다음 일봉 마감
```

여유가 매우 크다. 매시간 실행(`--interval 3600`)하는 것은 안전 마진일 뿐,
23시간을 꺼놨다가 하루 한 번 켜도 결과는 같다.

> 주식으로 한다면 창이 더 좁다. 장 마감(15:30) 후 ~ 다음날 개장(09:00) 전에
> 실행돼야 한다.

### 봉을 놓치면 어떻게 되는가

**놓친 봉은 소급 체결하지 않는다.** 과거 가격으로 되돌려 사고파는 것은 페이퍼
트레이딩이 아니라 백테스트이기 때문이다. 대신 **몇 개를 놓쳤는지 기록**한다.

```
[2026-03-05T09:12:00+00:00] * 처리 5 · 체결 2 · 평가 312,400원 · 놓친봉 15
```

`paper-report` 는 결손률이 10%를 넘으면 **비교 자체를 거부**한다.

```
  놓친 봉 45개 (전체의 33%) — 서버가 꺼져 있던 구간

  ⚠ 놓친 봉이 10%를 넘어 비교를 낼 수 없다.
    백테스트는 그 구간에도 매매한 것으로 계산하므로 같은 실험이 아니다.
    가동률부터 고치고 다시 시작할 것.
```

백테스트는 매 봉 매매한 것으로 계산하는데 라이브만 구멍이 뚫려 있으면
**같은 실험이 아니다.** 3개월 뒤 "왜 다르지?"를 고민하게 되므로, 결손이 크면
차라리 다시 시작하는 편이 낫다.

### 어디서 돌릴 것인가

| 방식 | 월 비용 | 가동률 | 비고 |
|---|---|---|---|
| **VPS** (최소 사양) | 5,000~8,000원 | ~100% | 가장 확실. 신경 쓸 일 없음 |
| **라즈베리파이** | 전기료 1,000원 미만 | 높음 | 3~5W. 집 인터넷·정전에 의존 |
| **집 PC 상시 가동** | 전기료 5,000~15,000원 | 중간 | 소음·수명. 절전 해제 필수 |
| **집 PC 하루 한 번** | 0원 | 낮음 | 켜는 걸 잊으면 결손 |
| 노트북 덮개 닫기 | — | **매우 낮음** | 절전 들어가면 실행 안 됨 |

**추천: 라즈베리파이 또는 최소 사양 VPS.** 3개월 총비용이 커피 두어 잔이고,
"켜는 걸 잊었다"는 실패 모드가 사라진다.

### 집 PC로 할 때 반드시 할 것

**절전이 최대의 적이다.** 절전 상태에서는 cron 도 systemd timer 도 돌지 않는다.

**Linux**
```bash
sudo systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
```

**macOS** — `caffeinate` 로 감싸거나 시스템 설정에서 잠자기 해제
```bash
sudo pmset -a sleep 0 disksleep 0
```

**Windows** — 전원 옵션에서 절전 안 함 + 작업 스케줄러에서
"작업을 실행하기 위해 절전 모드 해제" 체크

### 하루 한 번만 켜는 운용

PC를 매일 켜긴 하는데 상시 가동은 싫다면, 부팅 직후 1회 실행이면 충분하다.

```bash
# crontab -e   (@reboot 은 부팅 시 1회)
@reboot sleep 60 && cd /home/me/auto_trading && .venv/bin/python -m quant paper     -c configs/experiment_coin.yaml --best-file reports/coin/optimization_best.json     --name coin3m >> logs/paper.log 2>&1
```

systemd timer 를 쓴다면 `Persistent=true` 덕분에 **꺼져 있던 동안 놓친 실행을
부팅 후 자동 보충**한다(`deploy/quant-paper.timer`). cron 에는 없는 기능이다.

> 다만 보충되는 것은 *실행*이지 *놓친 봉*이 아니다. 3일 꺼놨다가 켜면
> 한 번 실행되고 최신 봉으로만 판단한다. 결손 3봉은 그대로 기록된다.

---

## 4. 모니터링

### 상태 확인

```bash
python -m quant paper-status -c configs/experiment_coin.yaml \
    --best-file reports/coin/optimization_best.json --name coin3m
```

```
페이퍼 계좌 현황
================================================================
  투입원금             300,000원  (초기 100,000 + 적립 200,000)
  평가금액             346,595원
  손익                 +46,595원  (+15.53%)
  누적수수료               512원
  체결 건수                 12건

  보유 종목
    종목                  수량        평가금액      비중
    KRW-BTC       0.00123456       121,122   34.9%
    KRW-ETH       0.04872833       135,303   39.0%
```

### 죽었는지 감시

```bash
./deploy/healthcheck.sh state/coin3m.jsonl
# 0 9 * * * /home/me/auto_trading/deploy/healthcheck.sh  (cron 하루 1회)
```

마지막 저널 갱신이 3시간(`MAX_AGE_MIN`) 넘게 없으면 CRITICAL 을 낸다.
**조용히 죽어 있는 것이 가장 흔한 실패 모드다.** 3개월 뒤에 열어봤더니 2주차에
멈춰 있었다는 사고를 막아준다.

---

## 5. 3개월 뒤 — 판정

```bash
python -m quant paper-report -c configs/experiment_coin.yaml \
    --best-file reports/coin/optimization_best.json --name coin3m
```

```
라이브 vs 백테스트
================================================================
  지표                라이브        백테스트           차이
  ------------------------------------------------------
  총수익률           +15.65%       +14.62%        +1.02%
  CAGR              +82.31%       +75.40%        +6.91%
  연변동성           +48.22%       +47.10%        +1.12%
  Sharpe              1.412         1.386         0.026
  MDD               -19.55%       -22.04%        +2.49%

  운영일수 90일 · 체결 12건 · 수수료 부담 0.17%

  ✓ 라이브와 백테스트가 비슷하다. 백테스트 가정이 현실적이었다는 뜻.
```

### 판정 기준

| 상황 | 해석 | 다음 행동 |
|---|---|---|
| 라이브 ≈ 백테스트 | **가정이 현실적이었다** | 소액 실전 검토 가능 |
| 라이브 < 백테스트 (10%p 초과) | 슬리피지·체결 가정이 낙관적이었다 | 비용 올려 재최적화 |
| 라이브 ≫ 백테스트 | 운이 좋았거나 **버그** | 의심할 것. 3개월은 짧다 |
| 운영일수 < 30일 | 표본 부족 | **판단하지 말 것** |

> **가장 중요한 주의.** 3개월 90봉은 통계적으로 매우 작은 표본이다.
> 이 기간의 수익률은 대부분 **시장 방향**이지 전략 실력이 아니다.
> 상승장이면 아무 전략이나 벌고, 하락장이면 다 잃는다.
>
> 그래서 봐야 할 것은 수익률이 아니라 **괴리**다. 라이브와 백테스트가
> 비슷하게 움직였다면, 그것만으로 이 3개월은 성공이다.

### 반드시 함께 볼 것

```bash
# 벤치마크(그냥 사서 들고 있기)와 비교
python -m quant backtest -c configs/experiment_coin.yaml -s buy_and_hold
```

전략이 `buy_and_hold` 를 못 이겼다면, 3개월간 컴퓨터를 돌린 대가가 마이너스다.

---

## 6. 3개월 운영 체크리스트

### 시작 전
```
□ WFE ≥ 0.5 확인 (또는 "운영 검증이 목적"임을 명시적으로 인정)
□ python -m pytest tests/ -q 통과
□ check-data 로 업비트 시세 수집 확인
□ 서버 시간대 Asia/Seoul
□ systemd enable (재부팅 후 자동 시작) 확인 — 실제로 리부팅해서 테스트할 것
□ 디스크 여유 2GB
□ healthcheck cron 등록
```

### 매주
```
□ paper-status 로 잔고·보유 확인
□ journalctl 에서 error/cycle_error 이벤트 확인
□ 체결이 아예 없다면 원인 파악 (신호가 안 나오는가, 최소주문에 걸리는가)
```

### 절대 하지 말 것
```
✗ 운영 중 전략·파라미터 변경  — 3개월 실험이 무의미해진다. 바꾸려면 처음부터 다시
✗ 손실 났다고 중간에 끄기      — 그 자체가 실험 결과다
✗ state/*.json 수동 편집       — 손익 계산이 깨진다
✗ 3개월 성과로 실전 규모 결정   — 표본이 너무 작다
```

### 전략을 바꾸고 싶다면

끄고 고치지 말고 **다른 이름으로 하나 더 띄워라.** 여러 개를 동시에 돌리면
같은 기간·같은 시장에서 공정하게 비교된다.

```bash
python -m quant paper -c ... --name coin3m-grid   --strategy grid -p band=0.2 --loop &
python -m quant paper -c ... --name coin3m-trend  --strategy sma_cross -p fast=20 -p slow=60 --loop &
```

---

## 7. 안전 설계 — 왜 실제 주문이 불가능한가

이 모듈은 **주문 API 자체를 갖고 있지 않다.** 실수로라도 실주문이 나갈 수 없다.

| 보장 | 방법 |
|---|---|
| 주문 코드 부재 | `quant/live/` 어디에도 브로커 주문 호출이 없다 (테스트로 강제) |
| 인증정보 불요 | 업비트 **공개 시세 API**만 쓴다. API 키를 넣을 곳이 없다 |
| 상태 격리 | 가상 잔고는 `state/*.json` 에만 존재 |
| 킬 스위치 | `touch STOP` → 다음 사이클에 스스로 종료 |

```python
# tests/test_paper_trading.py
def test_no_real_orders_are_possible():
    src = Path(portfolio.__file__).read_text() + Path(runner.__file__).read_text()
    for banned in ("order-cash", "requests.post", "api_key", "APP_SECRET"):
        assert banned not in src
```

---

## 8. 알려진 차이 — 라이브가 백테스트와 완전히 같지는 않다

정직하게 밝혀둔다. 아래는 **설계상의 차이**이며 버그가 아니다.

**① 자본 배분 구조**
백테스트는 종목별 독립 계좌를 만들어 동일비중으로 합성한다. 라이브는 계좌가 하나다.
이를 맞추기 위해 라이브는 종목별 목표비중에 `1/N` 을 곱한다. 그래도 현금이 종목 간에
공유되므로 미세한 차이가 남는다(리허설에서 90일 기준 약 1%p).

**② 체결 시각**
백테스트는 "다음 봉 시가"에 체결한다. 라이브는 "봉 마감을 감지한 시점의 현재가"에
체결한다. `--interval` 이 짧을수록 가까워진다.

**③ 슬리피지**
백테스트는 고정 5bp 를 가정한다. 페이퍼 트레이딩도 **같은 가정을 쓴다** — 실제 호가창을
보지 않기 때문이다. 즉 **페이퍼 트레이딩은 슬리피지를 검증하지 못한다.**
이것만은 실전 소액으로만 확인할 수 있다.

**④ 부분 체결·거래정지 없음**
가상 체결은 항상 전량 즉시 체결된다.

> 요약: 페이퍼 트레이딩이 검증하는 것은 **전략 로직 · 운영 안정성 · 데이터 파이프라인**이다.
> **체결 품질은 검증하지 못한다.** 그건 실전 소액의 몫이다.

---

## 9. 다음 단계

3개월 뒤 라이브 ≈ 백테스트가 확인됐다면 [`docs/LIVE_TRADING.md`](LIVE_TRADING.md) 의
2단계(증권사 계좌·API)로 넘어간다. 그때도 **소액부터** 시작하고, 실전 슬리피지를
실측해 백테스트 비용 가정을 갱신해야 한다.
