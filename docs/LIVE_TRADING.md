# 로컬 실전 운용 준비 — 단계별 작업 목록

이 문서는 **당신이 로컬에서 직접 해야 하는 작업**만 순서대로 정리한 것이다.

> **먼저 읽을 것.** "실시간"에는 두 가지 뜻이 있고, 필요한 작업량이 10배 차이난다.
>
> | | 뜻 | 이 프레임워크와의 관계 | 작업량 |
> |---|---|---|---|
> | **A. 매일 자동 실행** | 매일 장 마감 후 신호 산출 → 다음날 시가 주문 | **지금 전략 그대로 쓴다** | 며칠 |
> | **B. 장중 틱 스트리밍** | 초/분 단위 체결 데이터로 장중 매매 | **전략을 처음부터 다시 만들어야 한다** | 몇 주~ |
>
> 현재 9개 전략은 전부 **일봉** 기준이다. `t`일 종가로 신호를 만들어 `t+1`일 시가에
> 체결하는 모델이므로, A에는 틱 데이터가 **원리적으로 필요 없다.**
> B로 가려면 분봉 데이터 수집부터 시작해 전략·백테스트·최적화를 전부 다시 해야 한다.
>
> 아래는 **A 기준**이며, B로 확장하는 방법은 §7에 따로 적었다.

---

## 전체 로드맵

```
0단계  로컬 환경 구축            (30분)
1단계  실데이터로 백테스트·최적화  (1~2시간, 대부분 대기)
2단계  KIS 계좌 개설 + API 키 발급 (1~2일, 심사 대기)
3단계  모의투자로 주문 연동 검증   (2~3일)  ← 절대 건너뛰지 말 것
4단계  스케줄러 자동화            (반나절)
5단계  소액 실전 + 모니터링       (최소 1개월)
```

---

## 0단계. 로컬 환경 구축

### 0-1. 저장소 클론

```bash
git clone https://github.com/icedo724/auto_trading.git
cd auto_trading
git checkout claude/repository-name-project-belee9
```

### 0-2. 가상환경 (전역 파이썬 오염 방지)

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
# 실행 정책 오류가 나면:
# Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 0-3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 0-4. 동작 확인

```bash
python -m pytest tests/ -q        # 68 passed 가 나와야 정상
python -m quant list              # 전략 9종 출력
```

### 0-5. 데이터 소스 진단 ← **여기서 막히면 다음 단계 무의미**

```bash
python -m quant check-data
```

`fdr` 또는 `naver` 중 하나라도 `정상`이면 통과. 전부 실패하면:

| 증상 | 원인 | 해결 |
|---|---|---|
| `finance-datareader 미설치` | 패키지 누락 | `pip install finance-datareader` |
| `Max retries exceeded` | 방화벽/프록시 | 사내망이면 개인 네트워크에서 재시도 |
| `SSLError` | 사내 SSL 검사 | 회사 CA 인증서 설치 필요 |
| 전부 실패 | 네트워크 차단 | `source: synthetic` 으로 파이프라인만 검증 가능 |

---

## 1단계. 실데이터로 백테스트·최적화

### 1-1. 유니버스 설정

`configs/experiment_kr.yaml` 을 열어 `symbols` 를 원하는 종목으로 바꾼다.

```yaml
data:
  source: fdr
  symbols: ["005930", "000660", "035420", "035720", "051910"]
  start: "-8y"        # 상대 표기 지원. 스케줄 실행 시 구간이 따라 움직인다
  end:   "today"      # 고정 날짜로 두면 신호가 과거에 멈춘다 — 반드시 today
  min_bars: 250
```

> **유니버스 선정 시 반드시 지킬 것**
> - **10종목 이상**. 3~5종목이면 성과가 종목 운에 좌우된다.
> - **일평균 거래대금 100억 이상**. 소형주는 슬리피지 가정(5bp)이 전혀 안 맞는다.
> - **생존 편향 주의.** 지금 잘나가는 종목만 넣으면 과거 성과가 부풀려진다.
>   "8년 전에 내가 이 종목을 골랐을까?"를 자문할 것.

### 1-2. 시세 수집

```bash
python -m quant fetch -c configs/experiment_kr.yaml
```

`data/cache/fdr/` 에 종목별 CSV가 쌓인다. 이후 실행은 네트워크를 타지 않는다.

### 1-3. 최적화

```bash
python -m quant optimize -c configs/experiment_kr.yaml --top 20
```

`reports/kr/optimization.md` 와 `optimization_best.json` 이 생성된다.

### 1-4. 검증 ← **가장 중요한 단계**

```bash
python -m quant validate -c configs/experiment_kr.yaml
```

**워크포워드 효율(WFE)이 0.5 미만이면 여기서 멈춰야 한다.**
그리드 1위가 표본 잡음이라는 뜻이고, 실전에 넣으면 돈을 잃는다.
전략을 바꾸거나, 유니버스를 바꾸거나, 아이디어를 버려야 한다.

| WFE | 판정 | 다음 행동 |
|---|---|---|
| ≥ 1.0 | 견고 | 2단계 진행 |
| 0.5 ~ 1.0 | 쓸 만함 | 2단계 진행하되 소액으로 |
| 0 ~ 0.5 | 과최적화 | **중단.** 전략/유니버스 재검토 |
| < 0 | 폐기 | **중단.** 아이디어 자체를 버릴 것 |

파라미터 민감도도 함께 본다.

```bash
python -m quant sensitivity -c configs/experiment_kr.yaml -s donchian
```

특정 값에서만 점수가 뾰족하게 튀면 그 파라미터는 신뢰할 수 없다.

### 1-5. 신호 확인

```bash
python -m quant signal -c configs/experiment_kr.yaml \
    --best-file reports/kr/optimization_best.json --refresh
```

```
symbol       date     close  target_weight  prev_weight action
005930 2026-07-30   71200.0            1.0          0.0    BUY
000660 2026-07-30  198500.0            1.0          1.0   HOLD
035720 2026-07-30   45300.0            0.0          1.0   SELL
```

**여기까지가 데이터·전략 검증이다. 이 시점에서 수동으로 주문해도 전략은 이미 돌아간다.**
2단계부터는 주문 자동화이고, 없어도 운용은 가능하다.

---

## 2단계. 한국투자증권(KIS) 계좌 + API 키

증권사 Open API는 **계좌만 있으면 실시간 시세와 주문이 무료**다. 별도 데이터 이용료 없음.
한국투자증권을 권하는 이유: REST + WebSocket이라 OS를 안 가리고, **모의투자를 지원**한다.

### 2-1. 계좌 개설
- 한국투자증권 앱(한국투자)에서 비대면 계좌 개설 (신분증 + 본인 명의 계좌)
- 국내주식 거래 가능 계좌여야 한다

### 2-2. API 신청
1. [apiportal.koreainvestment.com](https://apiportal.koreainvestment.com) 회원가입
2. **KIS Developers 서비스 신청** → 계좌번호 연결
3. 승인까지 영업일 기준 시간이 걸릴 수 있다
4. 승인 후 **APP KEY / APP SECRET** 발급

### 2-3. 모의투자 계좌 신청 (필수)
- 모의투자는 **별도 신청**이며 APP KEY도 실전과 다르게 발급된다
- 모의투자 참가 신청 후 모의 계좌번호를 받는다

### 2-4. 키 보관 — 절대 코드에 넣지 말 것

저장소에 `.env.example` 템플릿이 들어 있다. 복사해서 채운다.

```bash
cp .env.example .env
```

```bash
# .env
KIS_APP_KEY=발급받은_앱키
KIS_APP_SECRET=발급받은_시크릿
KIS_ACCOUNT_NO=12345678-01
KIS_ENV=paper          # paper=모의투자, real=실전
```

`.env` 는 이미 `.gitignore` 에 있다. 확인:

```bash
git check-ignore -v .env      # .gitignore:7:.env  가 나오면 정상
```

**키가 유출되면 남이 내 계좌로 주문할 수 있다.** 스크린샷·블로그·깃허브 주의.

### 2-5. 엔드포인트 (2026년 기준 — **반드시 공식 문서에서 재확인**)

| 항목 | 실전 | 모의투자 |
|---|---|---|
| REST | `https://openapi.koreainvestment.com:9443` | `https://openapivts.koreainvestment.com:29443` |
| WebSocket | `ws://ops.koreainvestment.com:21000` | `ws://ops.koreainvestment.com:31000` |

| 기능 | 경로 | TR ID (실전 / 모의) |
|---|---|---|
| 접근토큰 발급 | `POST /oauth2/tokenP` | — |
| 웹소켓 접속키 | `POST /oauth2/Approval` | — |
| 현재가 조회 | `GET /uapi/domestic-stock/v1/quotations/inquire-price` | `FHKST01010100` |
| 일봉 조회 | `GET /uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice` | `FHKST03010100` |
| 현금 매수 | `POST /uapi/domestic-stock/v1/trading/order-cash` | `TTTC0802U` / `VTTC0802U` |
| 현금 매도 | `POST /uapi/domestic-stock/v1/trading/order-cash` | `TTTC0801U` / `VTTC0801U` |
| 잔고 조회 | `GET /uapi/domestic-stock/v1/trading/inquire-balance` | `TTTC8434R` / `VTTC8434R` |

> ⚠️ **TR ID와 경로는 KIS가 예고 없이 바꾼다.** 위 표는 참고용이며,
> 주문 TR ID를 하나 잘못 쓰면 매수가 매도로 나가거나 주문이 실패한다.
> 반드시 [KIS 공식 API 문서](https://apiportal.koreainvestment.com)에서
> 현재 값을 확인하고, **모의투자에서 실제 응답을 찍어본 뒤** 코드에 넣을 것.

### 2-6. 알아둘 제약
- **접근토큰 유효기간 24시간.** 매번 재발급하면 안 되고 캐시해서 재사용해야 한다
  (짧은 시간에 반복 발급하면 차단된다)
- **호출 유량 제한** 존재. 실전과 모의의 한도가 다르다. 종목이 많으면 설계 필요
- **주문 시 hashkey** 가 필요한 경우가 있다 (`POST /uapi/hashkey`)
- 모의투자는 실제 체결과 다르다. **호가 갭·부분체결·거래정지가 재현되지 않는다**

---

## 3단계. 모의투자 연동 검증 — 건너뛰지 말 것

여기서 확인해야 할 것은 "전략이 돈을 버는가"가 아니라 **"주문이 의도대로 나가는가"** 다.

### 3-1. 단계별 체크리스트

```
□ 토큰 발급 성공, 24시간 캐시 동작
□ 잔고 조회 → 예수금/보유수량이 앱 화면과 일치
□ 1주 시장가 매수 → 앱에서 체결 확인
□ 1주 시장가 매도 → 앱에서 체결 확인
□ signal 출력의 BUY/SELL 이 실제 주문으로 정확히 매핑되는지
□ 같은 신호를 두 번 실행해도 주문이 두 번 안 나가는지 (중복 방지)
□ 장 마감 후 실행 시 주문이 거부되는지 (또는 예약주문으로 가는지)
□ 잔고 부족 / 없는 종목코드 / 네트워크 끊김 시 에러 처리
```

### 3-2. 반드시 넣어야 할 안전장치

| 장치 | 이유 |
|---|---|
| **킬 스위치** | 파일 하나(`STOP`) 있으면 주문 중단. 급할 때 코드 수정할 여유 없다 |
| **1회 주문 상한** | 종목당 최대 금액/수량. 계산 버그로 전 재산이 한 종목에 들어가는 사고 방지 |
| **일일 주문 횟수 상한** | 무한 루프 시 수수료로 계좌가 녹는 것 방지 |
| **중복 주문 방지** | 날짜+종목+방향 키로 이미 실행했는지 기록. 스케줄러 재실행 사고 대비 |
| **장 운영시간 체크** | 휴장일·주말에 주문 시도하지 않기 |
| **전량 로깅** | 요청/응답 원문을 파일에. 사고 나면 이것밖에 단서가 없다 |
| **드라이런 모드** | `--dry-run` 으로 주문 직전까지만 실행하고 출력만 |

### 3-3. 최소 2주는 모의투자로 돌린다

매일 신호가 나오고 주문이 나가는 것을 눈으로 확인한다.
이 기간에 발견되는 문제의 대부분은 전략이 아니라 **운영**이다 —
휴장일, 데이터 지연, 토큰 만료, 종목 거래정지 등.

---

## 4단계. 스케줄러 자동화

### 4-1. 실행 시각 설계

한국 증시는 09:00~15:30 (동시호가 15:20~15:30).
일봉 전략이므로 **당일 종가가 확정된 후** 신호를 만들고, **다음 거래일 시가**에 주문한다.

```
15:40  장 마감 후 → 데이터 수집 + 신호 산출 + 주문 계획 저장
08:50  다음날 개장 전 → 저장된 계획대로 주문 제출
```

> 15:40에 신호를 뽑아 즉시 주문하면 **백테스트 가정(다음날 시가 체결)과 어긋난다.**
> 백테스트에서 검증한 것과 다른 것을 실행하게 되므로, 성과가 재현되지 않는다.

### 4-2. 실행 스크립트 (저장소에 동봉)

`run_daily.sh`(macOS/Linux)와 `run_daily.bat`(Windows)이 이미 들어 있다.
킬 스위치·로그·최적파라미터 존재 확인이 포함되어 있으므로 그대로 쓰면 된다.

```bash
./run_daily.sh                                   # 기본 설정으로 실행
QUANT_CONFIG=configs/my.yaml ./run_daily.sh      # 설정 바꿔서 실행
touch STOP && ./run_daily.sh                     # 킬 스위치 동작 확인 (아무것도 안 함)
```

<details><summary>스크립트 내용 (직접 수정하고 싶다면)</summary>

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

LOG="logs/$(date +%Y-%m-%d).log"
mkdir -p logs reports/live

{
  echo "===== $(date '+%F %T') 신호 산출 ====="
  python -m quant signal \
      -c configs/experiment_kr.yaml \
      --best-file reports/kr/optimization_best.json \
      --refresh \
      --out reports/live
} >> "$LOG" 2>&1
```

```bash
chmod +x run_daily.sh   # 클론 직후 실행 권한이 없다면
```

**Windows** (`run_daily.bat`):

```bat
@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat
if not exist logs mkdir logs
python -m quant signal -c configs\experiment_kr.yaml ^
    --best-file reports\kr\optimization_best.json --refresh --out reports\live ^
    >> logs\daily.log 2>&1
```

</details>

### 4-3. 스케줄 등록

**macOS / Linux (cron)**

```bash
crontab -e
```
```cron
# 평일 15:40 신호 산출  (KST 기준. 서버 TZ 확인 필수: date 명령으로)
40 15 * * 1-5 /Users/me/auto_trading/run_daily.sh
```

> macOS는 `crontab` 대신 `launchd` 를 권장하며, **절전 중에는 cron이 안 돈다.**
> 노트북이라면 시스템 설정에서 절전 해제 시간을 잡거나 상시 켜두는 PC를 쓸 것.

**Windows (작업 스케줄러)**

```powershell
schtasks /create /tn "quant-daily" /tr "C:\auto_trading\run_daily.bat" ^
         /sc weekly /d MON,TUE,WED,THU,FRI /st 15:40
```

### 4-4. 주간 재최적화

파라미터는 시장이 변하면 낡는다. 주 1회 또는 월 1회 재최적화를 건다.

```cron
# 매주 토요일 02:00 재최적화 + 검증
0 2 * * 6 /Users/me/auto_trading/run_reoptimize.sh
```

> **주의:** 재최적화 결과를 자동으로 실전에 반영하지 말 것.
> WFE가 무너졌는지 사람이 보고 판단한 뒤 반영해야 한다.

### 4-5. 알림

신호와 주문 결과를 텔레그램/슬랙으로 받으면 매일 로그를 열지 않아도 된다.
텔레그램 봇이 가장 간단하다 (`@BotFather` → 토큰 → `sendMessage` API).

---

## 5단계. 실전 전환

### 5-1. 전환 조건

```
□ 모의투자 2주 이상 무사고 운영
□ WFE ≥ 0.5
□ 킬 스위치 동작 확인
□ 주문 상한 동작 확인
□ 휴장일 처리 확인
□ 최대 손실 한도를 스스로 정했고, 그 금액을 잃어도 괜찮은가
```

### 5-2. 소액부터

- **첫 달은 전체 투자금의 5~10%** 만. 백테스트 성과는 재현되지 않는 경우가 훨씬 많다
- 모의투자와 실전의 체결 차이(슬리피지·부분체결)를 실측한다
- 실측 슬리피지가 백테스트 가정(5bp)보다 크면 **비용을 올려 재최적화**해야 한다

### 5-3. 매달 점검

| 항목 | 확인 |
|---|---|
| 실현 슬리피지 | 백테스트 가정과 비교. 크면 `slippage_bps` 상향 |
| 실제 vs 백테스트 수익률 | 괴리가 크면 원인 규명 전까지 증액 금지 |
| MDD | 백테스트 MDD를 넘으면 즉시 중단하고 재검토 |
| 데이터 품질 | 액면분할·거래정지 종목이 섞이지 않았는지 |

---

## 6단계. 운영 중 자주 터지는 문제

| 증상 | 원인 | 대응 |
|---|---|---|
| 신호 날짜가 어제에서 안 바뀜 | `end` 가 고정 날짜 | `end: "today"` 로 변경 |
| 주말에 신호가 안 바뀜 | 정상 (휴장) | 평일만 스케줄 |
| 특정 종목만 데이터 없음 | 상장폐지·거래정지·코드 변경 | 유니버스에서 제외 |
| 갑자기 -50% 봉이 찍힘 | 액면분할 미반영 | 수정주가 소스 확인, 캐시 `--refresh` |
| 토큰 오류 반복 | 24시간 만료 or 과다 발급 | 토큰 캐시 구현 확인 |
| 주문은 나갔는데 미체결 | 지정가가 시장에서 멀다 | 시장가 또는 호가 기준 지정가 |
| 캐시가 옛날 데이터 반환 | 캐시 커버리지 판정 | `--refresh` 로 강제 갱신 |

---

## 7단계. (선택) 장중 틱 스트리밍으로 확장

**여기부터는 지금 전략을 그대로 쓸 수 없다.** 작업 순서만 적는다.

1. **분봉 데이터 확보** — KIS WebSocket(`H0STCNT0` 체결가)으로 실시간 수신하며
   1분봉으로 집계해 저장. 과거 분봉은 KIS REST로 제한적으로만 받을 수 있어,
   **직접 수집해 쌓아야 한다** (최소 수개월).
2. **전략 재작성** — 현재 9개 전략의 파라미터 단위가 "일"이다.
   분봉에 그대로 쓰면 의미가 달라진다. `trading_days` 도 분봉 기준으로 재정의해야 한다.
3. **비용 모델 재조정** — 회전율이 수십 배가 되므로 슬리피지가 성과를 지배한다.
   일봉의 5bp 가정은 전혀 맞지 않는다.
4. **재최적화·재검증** — 분봉 데이터로 WFE를 다시 확인.
5. **인프라** — 웹소켓 재연결, 장애 복구, 상시 가동 서버(노트북 불가).

> 현실적인 조언: **일봉 전략이 실전에서 1년 이상 살아남은 뒤에 고민할 문제다.**
> 장중 매매는 비용·인프라·심리 부담이 모두 커지는데, 일봉에서 못 이기는 아이디어가
> 분봉에서 이기는 경우는 드물다.

---

## 부록. 지금 당장 할 수 있는 최소 경로

주문 자동화 없이도 **오늘부터** 운용할 수 있다.

```bash
# 1) 환경
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m quant check-data

# 2) 종목 설정 후 최적화 + 검증
python -m quant optimize -c configs/experiment_kr.yaml
python -m quant validate -c configs/experiment_kr.yaml     # WFE 확인!

# 3) 매일 15:40 신호 확인 → MTS 앱에서 수동 주문
python -m quant signal -c configs/experiment_kr.yaml \
    --best-file reports/kr/optimization_best.json --refresh
```

종목이 10개 내외라면 수동 주문도 하루 1~2분이면 끝난다.
**자동 주문은 편의 기능이지 수익의 원천이 아니다.**
2~5단계는 전략이 실제로 작동한다는 확신이 선 뒤에 해도 늦지 않다.
