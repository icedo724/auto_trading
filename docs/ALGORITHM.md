# 알고리즘 수식 정의

이 문서는 `quant/` 구현을 수식으로 옮긴 것이다. 각 절 끝에 대응 소스 위치를 표기했다.

---

## 0. 표기법

거래일을 $t = 1, 2, \dots, N$ 로 두고, 종목 $i$ 의 일봉을 다음과 같이 쓴다.

$$O_t,\; H_t,\; L_t,\; C_t,\; V_t \quad (\text{시가, 고가, 저가, 종가, 거래량})$$

| 기호 | 의미 | 설정 항목 |
|---|---|---|
| $w_t \in [-1, 1]$ | 전략이 산출한 **목표 비중** (자본 대비) | — |
| $\tilde w_t$ | 지연·클리핑된 **실행 목표 비중** | — |
| $\ell$ | 신호 지연 (거래일) | `signal_lag` $\ge 1$ |
| $q_t$ | 보유 주식 수 | — |
| $B_t$ | 현금 | — |
| $E_t$ | 평가자산 $= B_t + q_t C_t$ | — |
| $c$ | 편도 수수료율 | `commission_bps` |
| $\kappa$ | 매도 거래세율 | `sell_tax_bps` |
| $s$ | 슬리피지율 | `slippage_bps` |
| $\tau$ | 리밸런싱 임계치 | `rebalance_threshold` |
| $\lambda,\ \lambda_{\mathrm{tr}},\ \theta,\ D$ | 손절·추적손절·익절률, 최대보유일 | `stop_loss_pct` 등 |
| $\mathbb{1}\{\cdot\}$ | 지시함수 | — |

관례상 $x_t = \text{NaN}$ 인 구간(지표 미형성)은 $w_t = 0$ (현금)으로 사상한다.
이는 `Strategy._finalize` 가 담당한다.

---

## 1. 지표

> `quant/indicators.py`

**단순이동평균** — 최근 $n$ 개가 모두 존재할 때만 정의된다.

$$\mathrm{SMA}_n(x)_t = \frac{1}{n}\sum_{k=0}^{n-1} x_{t-k}$$

**지수이동평균** — `adjust=False`, 감쇠계수 $\alpha = \dfrac{2}{n+1}$

$$\mathrm{EMA}_n(x)_t = \alpha x_t + (1-\alpha)\,\mathrm{EMA}_n(x)_{t-1}$$

**표준편차 / z-score** — 모집단 표준편차($\mathrm{ddof}=0$)를 쓴다.

$$\sigma_n(x)_t = \sqrt{\frac{1}{n}\sum_{k=0}^{n-1}\bigl(x_{t-k} - \mathrm{SMA}_n(x)_t\bigr)^2},
\qquad
z_{n,t} = \frac{x_t - \mathrm{SMA}_n(x)_t}{\sigma_n(x)_t}$$

**RSI** (Wilder) — 상승분/하락분을 $\alpha = 1/n$ 의 EMA로 평활한다.

$$
U_t = \max(C_t - C_{t-1},\, 0), \qquad
D_t = \max(C_{t-1} - C_t,\, 0)
$$

$$
\bar U_t = \tfrac{1}{n}U_t + \bigl(1-\tfrac{1}{n}\bigr)\bar U_{t-1},
\qquad
\bar D_t = \tfrac{1}{n}D_t + \bigl(1-\tfrac{1}{n}\bigr)\bar D_{t-1}
$$

$$
\mathrm{RSI}_{n,t} = 100 - \frac{100}{1 + \bar U_t / \bar D_t}
= 100 \cdot \frac{\bar U_t}{\bar U_t + \bar D_t}
\qquad (\bar D_t = 0 \Rightarrow \mathrm{RSI} = 100)
$$

**MACD**

$$
\mathrm{MACD}_t = \mathrm{EMA}_{n_f}(C)_t - \mathrm{EMA}_{n_s}(C)_t,
\qquad
\mathrm{Sig}_t = \mathrm{EMA}_{n_\sigma}(\mathrm{MACD})_t
$$

$$\mathrm{Hist}_t = \mathrm{MACD}_t - \mathrm{Sig}_t$$

**볼린저 밴드**

$$
\mathrm{BB}^{\pm}_t = \mathrm{SMA}_n(C)_t \pm m\,\sigma_n(C)_t,
\qquad
\mathrm{BB}^{0}_t = \mathrm{SMA}_n(C)_t
$$

**True Range / ATR**

$$
\mathrm{TR}_t = \max\bigl(H_t - L_t,\; |H_t - C_{t-1}|,\; |L_t - C_{t-1}|\bigr)
$$

$$
\mathrm{ATR}_{n,t} = \tfrac{1}{n}\mathrm{TR}_t + \bigl(1-\tfrac{1}{n}\bigr)\mathrm{ATR}_{n,t-1}
$$

**돈치안 채널** — **당일을 제외한** 과거 $n$ 일 기준이다 ($\mathrm{shift}(1)$).
당일 고가를 포함하면 $C_t > U_{n,t}$ 가 원리적으로 불가능해져 돌파 신호가 사라진다.

$$
U_{n,t} = \max_{1 \le k \le n} H_{t-k},
\qquad
L_{n,t} = \min_{1 \le k \le n} L_{t-k}
$$

**실현변동성** (연율화, 거래일 $T_y$)

$$
\hat\sigma_{n,t} = \sqrt{T_y}\;\cdot\;
\sigma_n\!\left(\frac{C_t}{C_{t-1}} - 1\right)_{\!t}
$$

**모멘텀** — 최근 $j$ 일을 건너뛴 $n$ 일 수익률 (단기 반전 회피)

$$
\mathrm{MOM}_{n,j,t} = \frac{C_{t-j}}{C_{t-j-n}} - 1
$$

---

## 2. 전략: 가격 → 목표 비중

> `quant/strategy/trend.py`, `quant/strategy/reversion.py`

모든 전략은 사상 $\;\{O,H,L,C,V\}_{1:t} \mapsto w_t\;$ 이며, **$t$ 시점 종가까지의 정보만** 쓴다.
아래에서 $\mathcal{F}_t = \mathbb{1}\{C_t > \mathrm{SMA}_{n_{\mathrm{tf}}}(C)_t\}$ 는 선택적 추세 필터
($n_{\mathrm{tf}} = 0$ 이면 $\mathcal{F}_t \equiv 1$).

### 2.1 무상태(stateless) 전략

신호가 조건식으로 직접 결정된다.

**buy_and_hold** (벤치마크)

$$w_t = 1$$

**sma_cross** — 파라미터 $(n_f, n_s, n_{\mathrm{tf}})$, $n_f < n_s$

$$
w_t = \mathbb{1}\bigl\{\mathrm{SMA}_{n_f}(C)_t > \mathrm{SMA}_{n_s}(C)_t\bigr\}\cdot \mathcal{F}_t
\;-\;
\underbrace{\mathbb{1}\bigl\{\mathrm{SMA}_{n_f}(C)_t < \mathrm{SMA}_{n_s}(C)_t\bigr\}\cdot \bar{\mathcal{F}}_t}_{\text{allow\_short 일 때만}}
$$

여기서 $\bar{\mathcal{F}}_t = \mathbb{1}\{C_t < \mathrm{SMA}_{n_{\mathrm{tf}}}(C)_t\}$.

**macd** — 파라미터 $(n_f, n_s, n_\sigma, n_{\mathrm{tf}})$

$$
w_t = \mathbb{1}\{\mathrm{Hist}_t > 0\}\cdot\mathbb{1}\{\mathrm{MACD}_t > 0\}\cdot\mathcal{F}_t
$$

히스토그램 부호만으로는 하락 추세의 반등에도 켜지므로 $\mathrm{MACD}_t > 0$ 을 함께 요구한다.

**momentum** — 파라미터 $(n, j, \eta, n_{\mathrm{tf}})$

$$
w_t = \mathbb{1}\bigl\{\mathrm{MOM}_{n,j,t} > \eta\bigr\}\cdot\mathcal{F}_t
$$

**vol_target** — 파라미터 $(n_{\mathrm{ma}}, n_v, \sigma^\ast, w_{\max})$. 유일하게 **연속적 비중**을 낸다.

$$
w_t =
\Bigl\lfloor
\frac{1}{\delta}\min\Bigl(\frac{\sigma^\ast}{\hat\sigma_{n_v,t}},\, w_{\max}\Bigr)
\Bigr\rceil \cdot \delta
\;\cdot\;
\mathbb{1}\bigl\{C_t > \mathrm{SMA}_{n_{\mathrm{ma}}}(C)_t\bigr\},
\qquad \delta = 0.1
$$

$\lfloor\cdot\rceil$ 는 반올림. $\delta = 0.1$ 양자화는 미세 비중 변화로 인한 과다 회전을 막는다.
변동성이 커지면 노출이 자동으로 줄어 MDD가 개선된다.

### 2.2 상태 보유(stateful) 전략

진입/청산 조건이 **분리**되어 있어, 한 번 진입하면 청산 조건이 뜰 때까지 보유한다.
진입 조건 $a_t$, 청산 조건 $x_t$ 에 대해 상태 재귀는

$$
w_t = \mathcal{H}(a, x)_t =
\begin{cases}
1 & w_{t-1} = 0 \;\wedge\; a_t\\[2pt]
0 & w_{t-1} = 1 \;\wedge\; x_t\\[2pt]
w_{t-1} & \text{그 외}
\end{cases}
\qquad w_0 = 0
$$

이 재귀가 없으면 조건식이 참인 날에만 보유하게 되어 하루 단위로 진입/청산이 반복된다
(`_hold_between`, `DonchianBreakout.generate_signals`).

**donchian** — 파라미터 $(n_{\mathrm{in}}, n_{\mathrm{out}}, n_a, \alpha_{\max})$, $n_{\mathrm{out}} \le n_{\mathrm{in}}$

$$
a_t = \mathbb{1}\{C_t > U_{n_{\mathrm{in}},t}\}\cdot
\underbrace{\mathbb{1}\Bigl\{\tfrac{\mathrm{ATR}_{n_a,t}}{C_t} < \alpha_{\max}\Bigr\}}_{\alpha_{\max} > 0 \text{ 일 때만}},
\qquad
x_t = \mathbb{1}\{C_t < L_{n_{\mathrm{out}},t}\}
$$

**rsi_reversion** — 파라미터 $(n, \underline{r}, \bar{r}, n_{\mathrm{tf}})$, $\underline{r} < \bar{r}$

$$
a_t = \mathbb{1}\{\mathrm{RSI}_{n,t} < \underline{r}\}\cdot\mathcal{F}_t,
\qquad
x_t = \mathbb{1}\{\mathrm{RSI}_{n,t} > \bar{r}\}
$$

**bollinger** — 파라미터 $(n, m, \mathrm{mode}, n_{\mathrm{tf}})$

$$
(a_t, x_t) =
\begin{cases}
\bigl(\mathbb{1}\{C_t < \mathrm{BB}^-_t\}\cdot\mathcal{F}_t,\;\; \mathbb{1}\{C_t > \mathrm{BB}^0_t\}\bigr)
& \mathrm{mode} = \texttt{reversion}\\[4pt]
\bigl(\mathbb{1}\{C_t > \mathrm{BB}^+_t\}\cdot\mathcal{F}_t,\;\; \mathbb{1}\{C_t < \mathrm{BB}^0_t\}\bigr)
& \mathrm{mode} = \texttt{breakout}
\end{cases}
$$

**zscore** — 파라미터 $(n, z_{\mathrm{in}}, z_{\mathrm{out}})$, $z_{\mathrm{in}} < z_{\mathrm{out}}$

$$
a_t = \mathbb{1}\{z_{n,t} < z_{\mathrm{in}}\}\cdot\mathcal{F}_t,
\qquad
x_t = \mathbb{1}\{z_{n,t} > z_{\mathrm{out}}\}
$$

### 2.3 워밍업

각 전략은 지표 형성에 필요한 최소 봉 수 $\mathcal{W}$ 를 선언한다. 이 값이
**동일 시점 비교**의 기준이 된다(§6.1).

| 전략 | $\mathcal{W}$ |
|---|---|
| `sma_cross` | $\max(n_s,\, n_{\mathrm{tf}}) + 1$ |
| `macd` | $\max(n_s + n_\sigma,\, n_{\mathrm{tf}}) + 1$ |
| `donchian` | $\max(n_{\mathrm{in}},\, n_{\mathrm{out}},\, n_a) + 1$ |
| `momentum` | $n + j + n_{\mathrm{tf}} + 1$ |
| `rsi_reversion` | $\max(3n,\, n_{\mathrm{tf}}) + 1$ |
| `bollinger`, `zscore` | $\max(n,\, n_{\mathrm{tf}}) + 1$ |
| `vol_target` | $\max(n_{\mathrm{ma}},\, n_v) + 1$ |

RSI가 $3n$ 인 것은 EMA가 무한 메모리이기 때문이다. $n$ 봉에서는 초기값 편향이 남는다.

---

## 3. 체결과 회계

> `quant/engine.py`

### 3.1 신호 지연과 클리핑

$$
\tilde w_t = \mathrm{clip}\bigl(w_{t-\ell},\; -\bar w \cdot \mathbb{1}\{\text{allow\_short}\},\; \bar w \bigr),
\qquad \ell \ge 1
$$

$\ell \ge 1$ 은 설정 검증 단계에서 강제된다. $\ell = 0$ 이면 $t$ 일 종가로 만든 신호를
$t$ 일에 체결하는 **룩어헤드 바이어스**가 된다.

### 3.2 체결 가격

기준가 $X_t$ 와 슬리피지 반영 체결가 $F_t$ 는

$$
X_t =
\begin{cases}
O_t & \texttt{execution} = \texttt{next\_open}\\
C_t & \texttt{execution} = \texttt{next\_close}
\end{cases}
\qquad
F_t = X_t\bigl(1 + s\cdot\mathrm{sgn}(\Delta_t)\bigr)
$$

매수는 위로, 매도는 아래로 불리하게 체결된다.

### 3.3 리밸런싱

체결 직전 자산과 현재 비중:

$$
E^{\mathrm{ex}}_t = B_{t-1} + q_{t-1}X_t,
\qquad
u_t = \frac{q_{t-1}X_t}{E^{\mathrm{ex}}_t}
$$

거래는 다음 조건에서만 발생한다 (미세 리밸런싱 억제):

$$
\bigl|\tilde w_t - u_t\bigr| \ge \tau
\quad\lor\quad
\bigl(\tilde w_t = 0 \;\wedge\; q_{t-1} \ne 0\bigr)
$$

이때 목표 주식 수와 주문량은

$$
q^\ast_t = \Bigl[\tilde w_t \frac{E^{\mathrm{ex}}_t}{X_t}\Bigr]_{\mathrm{trunc}},
\qquad
\Delta_t = q^\ast_t - q_{t-1}
$$

$[\cdot]_{\mathrm{trunc}}$ 는 `allow_fractional = false` 일 때 0 방향 절사(정수 주 단위).

거래비용과 현금 갱신:

$$
\phi_t = |\Delta_t|\,F_t\,\bigl(c + \kappa\,\mathbb{1}\{\Delta_t < 0\}\bigr),
\qquad
B_t = B_{t-1} - \Delta_t F_t - \phi_t
$$

거래세 $\kappa$ 는 **매도에만** 부과된다(국내 0.18%).

### 3.4 비용 포함 실효 단가

손익을 왜곡 없이 재기 위해 진입/청산가에 수수료를 흡수시킨다.

$$
p^{\mathrm{in}} = F(1 + c) \quad (\text{매수}),
\qquad
p^{\mathrm{out}} = F(1 - c - \kappa) \quad (\text{매도})
$$

추가 매수 시 평균 단가는 수량 가중으로 갱신된다.

$$
p^{\mathrm{in}} \leftarrow \frac{p^{\mathrm{in}} q_{t-1} + F(1+c)\,\Delta_t}{q_{t-1} + \Delta_t}
$$

라운드트립 수익률(롱):

$$
R = \frac{p^{\mathrm{out}}}{p^{\mathrm{in}}} - 1
$$

따라서 가격이 전혀 움직이지 않아도 한 번의 왕복매매에서 잃는 비용은

$$
\mathrm{Cost}_{\mathrm{RT}} = 1 - \frac{(1-s)(1-c-\kappa)}{(1+s)(1+c)}
$$

국내 기본값 $(c, \kappa, s) = (1.5, 18, 5)\,\mathrm{bp}$ 에서 $\mathrm{Cost}_{\mathrm{RT}} \approx 31\,\mathrm{bp}$ 다.
연 12회전 전략이면 비용만으로 연 3.7%가 사라진다 — §6.2의 회전율 벌점이 필요한 이유다.

### 3.5 장중 리스크 관리

보유 중이면 당일 고가·저가로 체결 여부를 판정한다. 롱 포지션의 고점 추적값은

$$
M_t = \max_{t_{\mathrm{in}} \le k \le t} H_k
$$

손절선은 고정 손절과 추적 손절 중 **먼저 걸리는(더 높은)** 쪽이다
(해당 파라미터가 0이면 그 항은 제외되고, 둘 다 0이면 손절 자체를 쓰지 않는다).

$$
S_t = \max\bigl(\underbrace{p^{\mathrm{in}}(1 - \lambda)}_{\lambda > 0},\;
\underbrace{M_t(1 - \lambda_{\mathrm{tr}})}_{\lambda_{\mathrm{tr}} > 0}\bigr)
$$

숏 포지션은 부호가 반전되어 $S_t = \min(\cdot)$, $M_t = \min_k L_k$ 가 된다.

체결 판정과 체결가(당일 레인지로 클리핑):

$$
\text{손절: } L_t \le S_t \;\Rightarrow\; F^{\mathrm{exit}} = \mathrm{clip}(S_t,\, L_t,\, H_t)
$$

$$
\text{익절: } H_t \ge p^{\mathrm{in}}(1+\theta) \;\Rightarrow\; F^{\mathrm{exit}} = \mathrm{clip}\bigl(p^{\mathrm{in}}(1+\theta),\, L_t,\, H_t\bigr)
$$

$$
\text{시간청산: } t - t_{\mathrm{in}} \ge D \;\Rightarrow\; F^{\mathrm{exit}} = C_t
$$

우선순위는 손절 → 익절 → 시간청산 (보수적 가정: 한 봉 안에서 손절과 익절이 모두
닿았다면 손절이 먼저 걸렸다고 본다).

**재진입 차단** — 강제청산 후에는 상태 $\beta_t = 1$ 이 되고, 신호가 0으로 리셋될 때까지
재진입하지 않는다. 이것이 없으면 손절 당일 신호가 여전히 1이라 즉시 재진입하는
무한 손절 루프가 생긴다.

$$
\beta_t = \mathbb{1}\{\text{강제청산 발생}\},
\qquad
\tilde w_t \leftarrow \tilde w_t \cdot \mathbb{1}\{\beta_{t-1} = 0\},
\qquad
\beta_t \leftarrow 0 \text{ if } \tilde w_t = 0
$$

### 3.6 종가 평가

$$
E_t = B_t + q_t C_t,
\qquad
\pi_t = \frac{q_t C_t}{E_t} \quad (\text{실제 보유 비중})
$$

평가 시작 시점 $t_0$ 에서 모든 후보를 동일 초기자본으로 정규화한다.

$$
\hat E_t = \frac{E_t}{E_{t_0}} \cdot E^{(0)},
\qquad
r_t = \frac{\hat E_t}{\hat E_{t-1}} - 1
$$

---

## 4. 성과 지표

> `quant/metrics.py`. 평가 구간 길이 $N$, 연 거래일 $T_y$, 무위험수익률 $r_f$.

$$
\text{총수익률} = \frac{\hat E_N}{\hat E_1} - 1,
\qquad
\mathrm{CAGR} = \Bigl(\frac{\hat E_N}{\hat E_1}\Bigr)^{T_y/N} - 1
$$

**드로다운과 MDD**

$$
\mathrm{DD}_t = \frac{\hat E_t}{\max_{k \le t} \hat E_k} - 1,
\qquad
\mathrm{MDD} = \min_t \mathrm{DD}_t
$$

**Sharpe / Sortino** — 초과수익 $e_t = r_t - r_f/T_y$, 표본표준편차($\mathrm{ddof}=1$)

$$
\mathrm{Sharpe} = \frac{\bar e}{\mathrm{sd}(e)}\sqrt{T_y},
\qquad
\mathrm{Sortino} = \frac{\bar e}{\sqrt{\dfrac{1}{|\mathcal{D}|}\sum_{t \in \mathcal{D}} e_t^2}}\sqrt{T_y},
\quad \mathcal{D} = \{t : e_t < 0\}
$$

Sortino는 하방 편차만 벌점화하므로, 상승 변동성이 큰 추세 전략이 Sharpe에서 받는
부당한 감점을 보정한다.

**Calmar / Ulcer**

$$
\mathrm{Calmar} = \frac{\mathrm{CAGR}}{|\mathrm{MDD}|},
\qquad
\mathrm{UI} = \sqrt{\frac{1}{N}\sum_t \mathrm{DD}_t^2}
$$

Ulcer 지수는 하락의 **깊이와 지속기간**을 함께 반영한다. MDD 한 점만 보는 것보다
"오래 물려 있었는가"를 잘 잡는다.

**거래 통계** — 라운드트립 집합 $\mathcal{T} = \{(R_k, \Pi_k)\}$ ($\Pi_k$ 는 통화 단위 손익)

$$
\text{승률} p = \frac{|\{k : R_k > 0\}|}{|\mathcal{T}|},
\qquad
\mathrm{PF} = \frac{\sum_{\Pi_k > 0}\Pi_k}{-\sum_{\Pi_k \le 0}\Pi_k}
$$

$$
\mathrm{Payoff} = \left|\frac{\overline{R^+}}{\overline{R^-}}\right|,
\qquad
\text{기대값} = p\,\overline{R^+} + (1-p)\,\overline{R^-}
$$

**노출도 / 회전율 / 비용부담**

$$
\text{Exposure} = \frac{1}{N}\sum_t \min(|\pi_t|, 1),
\qquad
\text{Turnover} = T_y \cdot \frac{1}{N}\sum_t |\pi_t - \pi_{t-1}|,
\qquad
\text{CostDrag} = \frac{\sum_t \phi_t}{E^{(0)}}
$$

---

## 5. 포트폴리오 합성

> `quant/metrics.py::compute_portfolio_metrics`

종목 집합 $\mathcal{S}$ 에 대해 **동일비중 일별 리밸런싱**을 가정한다.

$$
r^{\mathcal{P}}_t = \frac{1}{|\mathcal{S}_t|}\sum_{i \in \mathcal{S}_t} r^{(i)}_t,
\qquad
E^{\mathcal{P}}_t = E^{(0)}\prod_{k \le t}\bigl(1 + r^{\mathcal{P}}_k\bigr)
$$

$\mathcal{S}_t$ 는 $t$ 시점에 상장되어 있는 종목만 포함한다(상장 전은 NaN → 제외).
$\S 4$ 의 모든 지표를 $r^{\mathcal{P}}$ 와 $E^{\mathcal{P}}$ 에 그대로 적용한다.

---

## 6. 파라미터 최적화

> `quant/optimizer.py`

### 6.1 동일 시점 제약 — 프레임워크의 핵심

후보 집합 $\mathcal{C} = \{(\text{전략}, \boldsymbol{\vartheta})\}$ 에 대해, **모든 후보의 매매 시작
시점을 하나로 통일**한다.

$$
t_0 = \max_{c \in \mathcal{C}} \mathcal{W}(c)
\qquad\Longrightarrow\qquad
\tilde w^{(c)}_t = 0 \quad \forall\, t < t_0,\;\; \forall\, c \in \mathcal{C}
$$

이 제약이 없으면 $\mathrm{SMA}(5)$ 후보는 5봉째부터, $\mathrm{SMA}(200)$ 후보는
200봉째부터 평가된다. 두 후보는 **서로 다른 시장 구간**을 겪게 되고, 측정된 성과차는
전략의 우열이 아니라 구간의 우열이 된다. 여기에 §3.6의 재정규화
($\hat E_{t_0} = E^{(0)}$, 모든 후보 공통)와 단일 `BacktestConfig` 공유가 더해져
비교 조건이 완전히 통제된다.

### 6.2 목적함수

기본값 `robust` 는 Sharpe를 뼈대로 세 개의 벌점을 곱한다.

$$
J(c) = \mathrm{Sharpe}(c)\;\cdot\;
\underbrace{\sqrt{\frac{\min(n_c,\,30)}{30}}}_{\text{표본 신뢰도}}\;\cdot\;
\underbrace{\frac{1}{1 + 3\max(|\mathrm{MDD}_c| - 0.20,\; 0)}}_{\text{MDD 벌점}}\;\cdot\;
\underbrace{\frac{1}{1 + 0.02\max(\mathrm{Turnover}_c - 12,\; 0)}}_{\text{회전율 벌점}}
$$

- **표본 신뢰도**: 거래 3건으로 만든 Sharpe 2.0은 $\sqrt{3/30} = 0.32$ 배로 할인되어
  0.63이 된다. 200건으로 만든 Sharpe 1.2(할인 없음)에 밀린다. 우연한 대박을
  최적해로 뽑는 것을 막는 장치다.
- **MDD 벌점**: 20%까지는 무벌점, 초과분에 선형 감점. MDD 50%면 $1/(1+0.9) = 0.53$ 배.
- **회전율 벌점**: 연 12회전(월 1회) 초과분에 감점. 슬리피지 가정 오차에 대한
  민감도를 낮춘다.

대안으로 $J = \mathrm{Sharpe},\ \mathrm{Sortino},\ \mathrm{Calmar},\ \mathrm{CAGR}$ 를 선택할 수 있다.

### 6.3 최소거래수 필터와 탐색

$$
\hat c = \arg\max_{c \in \mathcal{C}} J(c)
\quad \text{s.t.} \quad
n_c \ge n_{\min} \;\;\lor\;\; c \in \mathcal{B}
$$

$\mathcal{B}$ 는 벤치마크 집합(`buy_and_hold`). 거래가 적은 것이 정상이므로 필터에서 면제되며,
비교 기준으로 항상 리더보드에 남는다. 필터 탈락 후보는 $J = -\infty$ 로 밀린다.

탐색은 파라미터 격자의 데카르트 곱 전체를 훑는다. `validate()` 를 통과하지 못하는
조합(예: $n_f \ge n_s$)은 생성 단계에서 제외된다.

$$
\Theta = \prod_{j} \Theta_j,
\qquad
|\mathcal{C}| = \sum_{\text{전략}} \bigl|\{\boldsymbol{\vartheta} \in \Theta : \mathrm{valid}(\boldsymbol{\vartheta})\}\bigr|
$$

### 6.4 파라미터 민감도

파라미터 $j$ 의 값 $v$ 에 대한 주변 분포를 본다.

$$
\bar J_j(v) = \underset{c\,:\,\vartheta_j = v}{\mathrm{mean}}\; J(c),
\qquad
\mathrm{sd}_j(v) = \underset{c\,:\,\vartheta_j = v}{\mathrm{sd}}\; J(c)
$$

$\bar J_j$ 가 특정 $v$ 에서만 뾰족하면 과최적화 신호다. 이웃 값들도 고르게 높은
**평평한 영역(plateau)** 을 골라야 실전에서 재현된다.

---

## 7. 과최적화 검증

> `quant/validation.py`

### 7.1 IS / OOS 분할

분할 시점 $t_s$ 기준으로 학습 구간에서 상위 $K$ 개를 고르고, 검증 구간 성과를 잰다.
지표 워밍업은 과거 데이터로 확보하되 **매매는 $t_s$ 부터만** 한다.

$$
\mathcal{C}^\ast = \operatorname{top-}K_{c}\; J\bigl(c \,\big|\, t < t_s\bigr),
\qquad
\text{성과 열화} \;\; \Delta_m(c) = m^{\mathrm{OOS}}(c) - m^{\mathrm{IS}}(c)
$$

생존 판정:

$$
\mathrm{Survived}(c) = \mathbb{1}\bigl\{\mathrm{Sharpe}^{\mathrm{OOS}} > 0.3\bigr\}\cdot
\mathbb{1}\bigl\{\mathrm{CAGR}^{\mathrm{OOS}} > 0\bigr\}\cdot
\mathbb{1}\bigl\{n^{\mathrm{OOS}} \ge 3\bigr\}
$$

### 7.2 워크포워드

학습창 $T_{\mathrm{tr}}$, 검증창 $T_{\mathrm{te}}$ 를 $T_{\mathrm{te}}$ 씩 굴린다.
창 $k$ 의 구간을

$$
\mathcal{I}^{\mathrm{tr}}_k = [\,t_0 + (k-1)T_{\mathrm{te}},\;\; t_0 + (k-1)T_{\mathrm{te}} + T_{\mathrm{tr}}\,),
\qquad
\mathcal{I}^{\mathrm{te}}_k = [\,\sup\mathcal{I}^{\mathrm{tr}}_k,\;\; \sup\mathcal{I}^{\mathrm{tr}}_k + T_{\mathrm{te}}\,)
$$

로 두면, 매 창마다 파라미터를 **재선택**하고 그 다음 구간에만 적용한다.

$$
c^\ast_k = \arg\max_{c} J\bigl(c \,\big|\, \mathcal{I}^{\mathrm{tr}}_k\bigr),
\qquad
r^{\mathrm{OOS}}_t = r_t\bigl(c^\ast_k\bigr) \quad \text{for } t \in \mathcal{I}^{\mathrm{te}}_k
$$

검증 구간들은 서로 겹치지 않으므로, 이어붙인 $\{r^{\mathrm{OOS}}_t\}$ 는 하나의 연속된
운용 기록이 된다.

$$
E^{\mathrm{OOS}}_t = E^{(0)}\prod_{k \le t}\bigl(1 + r^{\mathrm{OOS}}_k\bigr)
$$

**워크포워드 효율(WFE)** — 실전 성과가 학습 성과의 몇 배로 재현되는가.

$$
\mathrm{WFE} = \frac{\mathrm{CAGR}\bigl(E^{\mathrm{OOS}}\bigr)}
{\dfrac{1}{K}\sum_{k=1}^{K}\mathrm{CAGR}^{\mathrm{IS}}_k}
\qquad
\begin{cases}
\ \approx 1 & \text{견고}\\
\ < 0.5 & \text{과최적화 의심}\\
\ < 0 & \text{폐기}
\end{cases}
$$

$\mathrm{WFE}$ 는 그리드 1위 성과가 **표본 잡음인지 실제 엣지인지**를 가르는 최종 관문이다.
리더보드 1위의 Sharpe가 아무리 높아도 이 값이 0.5를 넘지 못하면 실전 투입해선 안 된다.

---

## 부록 A. 합성 시세 생성 모형

> `quant/data/synthetic.py`. 네트워크 없이 파이프라인 전체를 검증하기 위한 소스.
> 종목 코드 → 시드 결정이므로 언제 돌려도 동일한 시세가 나온다.

**국면 전환** — 강세/약세/횡보 3상태 마르코프 체인 $R_t \in \{\mathrm{bull}, \mathrm{bear}, \mathrm{chop}\}$

$$
P =
\begin{pmatrix}
0.985 & 0.006 & 0.009\\
0.012 & 0.972 & 0.016\\
0.011 & 0.009 & 0.980
\end{pmatrix},
\qquad
(\mu_R) = (0.18,\, -0.16,\, 0.01),
\qquad
(m_R) = (0.85,\, 1.45,\, 1.00)
$$

**확률변동성** — 로그변동성의 AR(1). GARCH의 제곱수익률 되먹임과 달리 발산하지 않으면서
변동성 군집은 그대로 재현한다.

$$
\log\sigma_t = \mu_\sigma + \phi\bigl(\log\sigma_{t-1} - \mu_\sigma\bigr) + \eta\,\varepsilon_t,
\qquad \phi = 0.97,\;\; \eta = 0.12
$$

정상상태 분산 $v_\infty = \dfrac{\eta^2}{1 - \phi^2}$ 에 대해, 실현 **분산**을 목표치에 맞추려면
($\sigma_t$ 가 로그정규이므로 $\mathbb{E}[\sigma^2] = e^{2\mu_\sigma + 2v_\infty}$)

$$
\mu_\sigma = \log\sigma_{\mathrm{base}} - v_\infty
\qquad\Longrightarrow\qquad
\mathbb{E}[\sigma_t^2] = \sigma_{\mathrm{base}}^2
$$

$\mu_\sigma = \log\sigma_{\mathrm{base}} - v_\infty/2$ 로 두면 $\mathbb{E}[\sigma]$ 가 맞춰질 뿐
$\mathbb{E}[\sigma^2]$ 는 $e^{v_\infty}$ 배 부풀어, 실현변동성이 목표보다 13% 높게 나온다.

**팻테일 충격** — 1% 확률로 3배 정규충격이 겹친다.

$$
z_t = \mathrm{clip}\bigl(g_t + \mathbb{1}\{u_t < 0.01\}\cdot 3 h_t,\; -5,\; 5\bigr),
\qquad g_t, h_t \sim \mathcal{N}(0,1)
$$

**수익률과 가격**

$$
\log\frac{C_t}{C_{t-1}} = \mathrm{clip}\Bigl(\mu_{R_t}\Delta t - \tfrac{1}{2}\bigl(\sigma_t m_{R_t}\bigr)^2 + \sigma_t m_{R_t} z_t,\; -0.4,\; 0.4\Bigr)
$$

목표 연변동성 $\sigma_{\mathrm{ann}}$ 을 정확히 맞추기 위해 기준 변동성을 보정한다.
$\boldsymbol{\pi}$ 는 $P$ 의 정상분포.

$$
\sigma_{\mathrm{base}} = \frac{\sigma_{\mathrm{ann}}\sqrt{\Delta t}}
{\sqrt{\sum_R \pi_R m_R^2}\;\cdot\;\sqrt{\mathbb{E}[z^2]}},
\qquad
\mathbb{E}[z^2] = 1 + 0.01 \cdot 3^2 = 1.09
$$

이 보정 후 실측 연변동성은 목표 0.28에 대해 0.27~0.28로 수렴한다.

**시가 갭과 장중 레인지**

$$
O_t = C_{t-1}e^{0.35\,\sigma_t\,\xi_t},
\qquad
H_t = \max(O_t, C_t)\bigl(1 + \rho_t \nu^+_t\bigr),
\qquad
L_t = \min(O_t, C_t)\bigl(1 - \rho_t \nu^-_t\bigr)
$$

$\rho_t = 0.9\,\sigma_t|\zeta_t|$, $\nu^\pm_t \sim \mathcal{U}(0.3, 1)$.
결과적으로 일중 레인지 / |일간수익률| 비가 약 2.2로, 실제 주식(1.5~2.5)과 부합한다.
