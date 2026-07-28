# auto_trading — 자동 주식매매 알고리즘 백테스트 · 파라미터 최적화 프레임워크

여러 매매 전략과 그 파라미터 조합을 **완전히 동일한 시점·동일한 조건**에서 경쟁시켜
최적 알고리즘을 찾아내는 퀀트 연구 도구.

```bash
pip install -r requirements.txt
python -m quant optimize -c configs/experiment_demo.yaml   # 네트워크 없이 바로 실행 가능
```

구현의 **수식 정의**는 [`docs/ALGORITHM.md`](docs/ALGORITHM.md) — 지표 · 전략 · 체결회계 ·
성과지표 · 목적함수 · 검증 절차를 수식으로 옮긴 문서.

---

## 1. 왜 "동일 시점"이 핵심인가

파라미터 비교에서 가장 흔한 오류는 **후보마다 다른 구간을 평가하는 것**이다.
MA(5) 전략은 데이터 5일째부터, MA(200) 전략은 200일째부터 매매를 시작하면
둘은 서로 다른 시장을 겪는다. 짧은 파라미터가 유리한 장을 더 오래 경험하면
"성과 차이"가 아니라 "구간 차이"를 측정하게 된다.

이 프레임워크는 그 함정을 **구조적으로** 차단한다.

| 통제 항목 | 방법 |
|---|---|
| 평가 시작일 | 전 후보의 `warmup` 중 **최댓값**에서 일제히 매매 시작 (`common_trade_start`) |
| 거래일 달력 | 유니버스 전체를 공통 달력에 정렬, 불일치 시 `grid_search`가 거부 |
| 초기 자본 | 평가 시작 시점에 전 후보를 `initial_cash`로 재정규화 |
| 거래비용·체결 | 단일 `BacktestConfig`를 전 후보가 공유 |
| 룩어헤드 | 신호를 `signal_lag`(≥1)만큼 강제 지연, `signal_lag=0`은 설정 단계에서 거부 |

```
[최적화] 후보 385개 · 목적함수=robust
[최적화] 공통 평가 구간 2017-08-14 ~ 2025-12-31 (모든 후보 동일)
```

---

## 2. 구조

```
quant/
├── config.py        BacktestConfig · CostModel (kr/us/zero)
├── indicators.py    SMA/EMA/RSI/MACD/볼린저/ATR/돈치안/ADX/z-score
├── data/
│   ├── base.py      OHLCV 스키마 정규화 · 공통 달력 정렬
│   ├── synthetic.py 합성 시세 (네트워크 불필요, 종목코드 → 시드 고정)
│   ├── remote.py    yfinance(미국) · pykrx(국내)
│   └── csv_source.py CSV 소스 + 디스크 캐시
├── strategy/        전략 레지스트리 (@register 로 자동 등록)
├── engine.py        백테스트 엔진 (비용·슬리피지·손절/익절/추적손절)
├── metrics.py       CAGR/Sharpe/Sortino/MDD/Calmar/PF/회전율 ...
├── optimizer.py     그리드 탐색 · 목적함수 · 민감도 분석
├── validation.py    IS/OOS 분할 · 워크포워드
├── report.py        마크다운/CSV/PNG 리포트
└── cli.py           명령줄 진입점
```

---

## 3. 내장 전략 (9종)

| 이름 | 유형 | 핵심 아이디어 | 그리드 |
|---|---|---|---|
| `buy_and_hold` | 벤치마크 | 매수 후 보유 | 1 |
| `sma_cross` | 추세 | 단기/장기 이평 교차 | 32 |
| `macd` | 추세 | MACD 히스토그램 부호 | 81 |
| `donchian` | 돌파 | N일 신고가 돌파 / M일 신저가 청산 (터틀) | 24 |
| `momentum` | 추세 | lookback 수익률 > 문턱 | 48 |
| `rsi_reversion` | 회귀 | RSI 과매도 매수 → 회복 청산 | 144 |
| `bollinger` | 회귀/돌파 | 밴드 이탈 매수 → 중심선 청산 | 48 |
| `zscore` | 회귀 | 가격 z-score 진입/청산 문턱 | 72 |
| `vol_target` | 추세+리스크 | 이평 위에서만 보유, 비중 = 목표변동성/실현변동성 | 27 |

기본 그리드 합계 **385 조합**. `configs/*.yaml`의 `optimize.grids`로 덮어쓸 수 있다.

### 새 전략 추가

```python
# quant/strategy/my_strategy.py
from .base import Strategy, register
from .. import indicators as ind

@register
class MyStrategy(Strategy):
    name = "my_strategy"
    defaults   = {"period": 20, "threshold": 0.0}
    param_space = {"period": [10, 20, 40], "threshold": [0.0, 0.02]}

    def validate(self):                      # 무효 조합은 그리드에서 자동 제외
        if self["period"] < 2:
            raise ValueError("period 가 너무 짧습니다.")

    @property
    def warmup(self) -> int:                 # 지표에 필요한 최소 봉 수
        return self["period"] + 1

    def generate_signals(self, df):          # t행은 t일 종가까지의 정보만 사용
        mom = df["close"] / ind.sma(df["close"], self["period"]) - 1
        return self._finalize((mom > self["threshold"]).astype(float), df.index)
```

`quant/strategy/__init__.py`에 import 한 줄만 추가하면 CLI·최적화기가 자동 인식한다.

---

## 4. 사용법

```bash
python -m quant list                                        # 전략/파라미터 공간
python -m quant fetch     -c configs/experiment_kr.yaml     # 시세 수집·캐시
python -m quant backtest  -c ... -s sma_cross -p fast=20 -p slow=60
python -m quant optimize  -c ... --objective robust --top 20
python -m quant sensitivity -c ... -s donchian
python -m quant validate  -c ... --split 2022-01-01
python -m quant signal    -c ... --best-file reports/demo/optimization_best.json
```

### 실험 설정 (`configs/*.yaml`)

```yaml
data:
  source: krx                    # synthetic | krx | yahoo | csv
  symbols: ["005930", "000660"]
  start: "2015-01-01"
  end:   "2025-12-31"

backtest:                        # 전 후보가 공유하는 단일 조건
  initial_cash: 10000000
  cost: kr                       # 수수료 0.015%/편도 + 거래세 0.18%(매도) + 슬리피지 0.05%
  execution: next_open           # t일 신호 → t+1일 시가 체결
  signal_lag: 1                  # 룩어헤드 방지 (0은 거부됨)
  allow_fractional: false        # 국내는 정수 주 단위
  stop_loss_pct: 0.10

optimize:
  objective: robust
  min_trades: 20                 # 표본 부족 후보 배제 (벤치마크는 면제)
  strategies: [sma_cross, donchian, rsi_reversion, ...]
  grids:                         # 생략 시 전략 클래스의 기본 param_space 사용
    sma_cross: { fast: [5,10,20,30], slow: [40,60,90,120] }

validate:
  split_date: "2022-01-01"
  train_days: 504                # 워크포워드 학습창 (약 2년)
  test_days: 126                 # 워크포워드 검증창 (약 6개월)
```

---

## 5. 백테스트 엔진이 반영하는 현실

| 항목 | 처리 |
|---|---|
| 체결 시점 | t일 종가 신호 → **t+1일 시가**(또는 종가) 체결 |
| 슬리피지 | 매수는 불리하게 위로, 매도는 아래로 체결가 이동 |
| 수수료 | 매수·매도 양방향 |
| 거래세 | **매도 시에만** (국내 0.18%) |
| 손절/익절 | 당일 **고가·저가**로 장중 체결 판정, 체결가는 `[low, high]`로 클리핑 |
| 추적손절 | 진입 후 고점 갱신분 반영 |
| 재진입 방지 | 강제청산 후에는 신호가 0으로 리셋될 때까지 재진입 차단 |
| 미세 리밸런싱 | `rebalance_threshold` 미만의 비중 변화는 거래 생략 |
| 단주 | `allow_fractional: false`면 정수 주 단위 |
| 상장 전 구간 | 공통 달력 정렬 시 NaN 유지 → 매매 대상에서 제외 |

---

## 6. 과최적화 방어

그리드 1위는 "그 구간에서 가장 운이 좋았던 조합"일 수 있다. 세 겹으로 방어한다.

**① 목적함수 `robust` (기본값)**

```
score = Sharpe × 표본신뢰도 × MDD페널티 × 회전율페널티
        √(min(거래수,30)/30)   MDD 20% 초과분   연 12회전 초과분
```

거래 3건으로 만든 Sharpe 2.0과 200건으로 만든 Sharpe 1.2 중 후자를 고른다.

**② 파라미터 민감도**

```bash
python -m quant sensitivity -c configs/experiment_demo.yaml -s donchian
```

```
── donchian.entry ──
 value  n  score_mean  score_max  score_std  cagr_mean  mdd_mean
    80  6       0.485      0.592      0.092      0.030    -0.139
    55  6       0.468      0.552      0.080      0.031    -0.130
    20  4       0.399      0.449      0.036      0.029    -0.153
    40  6       0.300      0.404      0.092      0.020    -0.177
```

특정 값 하나만 뾰족하게 튀면 과최적화 신호. 이웃 값들도 고르게 좋은
**평평한 영역(plateau)** 을 골라야 실전에서 재현된다.

**③ IS/OOS + 워크포워드**

```bash
python -m quant validate -c configs/experiment_demo.yaml
```

워크포워드는 학습창을 굴리며 매 구간 파라미터를 **재선택**하고
그 다음 구간의 실적만 이어붙인다. 실제 운용에 가장 가까운 추정치다.

```
워크포워드 효율(WFE): 0.87    # 1.0 근처=견고, 0.5 미만=과최적화 의심, 음수=폐기
```

> 동봉된 합성 데이터 데모에서는 WFE가 음수로 나온다. 이는 버그가 아니라
> **의도된 결과**다 — 예측 가능한 구조가 거의 없는 시계열에서 그리드 1위는
> 표본 잡음이며, 워크포워드가 그것을 정확히 드러낸다.
> 실데이터로 돌렸을 때 WFE가 0.5를 넘지 못하면 같은 판단을 내려야 한다.

---

## 7. 산출물

`optimize` / `validate` 실행 시 `optimize.output_dir`에 저장된다.

| 파일 | 내용 |
|---|---|
| `optimization.md` | 실험 조건 · 리더보드 · 전략별 최고 · 최종 선택 · 민감도 |
| `optimization_leaderboard.csv` | 전 후보의 파라미터 + 전 지표 |
| `optimization_best.json` | 최적 파라미터 (→ `signal --best-file` 입력) |
| `optimization_equity.csv/.png` | 상위 후보 자산곡선 + 드로다운 |
| `validation.md` | IS/OOS 표 · 워크포워드 창별 선택 이력 · WFE |

---

## 8. 데이터 소스

| source | 대상 | 패키지 | 비고 |
|---|---|---|---|
| `synthetic` | — | 없음 | 국면전환 + 변동성군집 합성 시세. 종목코드로 시드 고정 → 항상 동일 |
| `krx` | 국내 주식 | `pykrx` | 6자리 코드 (`005930`) |
| `yahoo` | 미국/글로벌 | `yfinance` | 티커 (`AAPL`, `SPY`) |
| `csv` | 임의 | 없음 | `data/csv/<종목>.csv` |

`krx`/`yahoo`는 `data/cache/`에 자동 캐시되어 재실행 시 네트워크를 타지 않는다.
(`--refresh`로 강제 갱신)

---

## 9. 테스트

```bash
python -m pytest tests/ -q      # 52 passed
```

핵심 검증 항목:

- **룩어헤드 부재** — 미래 데이터를 잘라내도 과거 신호가 변하지 않음 (전 전략)
- **`signal_lag` 효력** — 마지막 봉에만 켜지는 신호는 수익을 만들 수 없음
- **warmup 충분성** — `warmup` 봉의 과거만으로도 신호가 원 경로로 수렴
- **동일 시점 보장** — 전 후보의 자산곡선이 같은 인덱스·같은 초기자본에서 시작
- **비용 정합성** — 무의미한 매매 반복 시 정확히 비용만큼 손실
- **손절/익절/추적손절** 체결가, **정수 주** 모드, **공매도** 방향
- **직렬 == 병렬** 결과 일치

---

## 10. 한계 (실전 투입 전 반드시 확인)

- **일봉 기준**이다. 장중 변동, 호가 스프레드, 부분 체결은 모델링하지 않는다.
- 슬리피지는 **고정 bps**다. 실제로는 유동성·주문 규모에 따라 달라진다.
- 손절 체결가는 `[low, high]` 범위로 가정한다. **갭 하락 시 실제 체결은 더 나쁠 수 있다.**
- 종목 유니버스는 고정이다. **생존 편향**(상장폐지 종목 누락)이 남아 있다.
- 동일비중 일별 리밸런싱 포트폴리오를 가정한다. 실제 운용의 자금 배분과 다르다.
- 백테스트 성과는 미래 수익을 보장하지 않는다. 실계좌 투입 전 반드시
  소액·모의투자로 신호와 체결을 검증할 것.
