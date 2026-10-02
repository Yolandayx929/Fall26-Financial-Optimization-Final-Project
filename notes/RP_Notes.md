# Risk Parity + Common Performance Evaluation —— 进度与决策记录

> 本人负责 topic 3 的 **risk parity** 组合，以及五个组合共用的 **performance evaluation**、
> **risk-contribution diagnostics** 和 **alignment check**。
> 只新增文件，不改动队友的 `estimation.py` / `portfolios.py` / `backtest.py`。

---

## 1. 新增文件

| 文件 | 内容 |
|---|---|
| `risk_parity.py` | `risk_parity_weights(Sigma)`（dense，Newton）、`risk_parity_closed_form(cov)`（single-factor，生产用）、`risk_parity_cvxpy`（仅交叉验证）、`rc_dispersion`、backtest constructor `risk_parity` |
| `backtesting_analysis.py` | `performance_metrics`、`performance_table`、`subperiod_table`、`drifted_turnover`、`cumulative_returns`、`drawdowns`、`risk_contributions`（通用工具，所有组合共用）、`estimate_covs`、`risk_concentration`、`check_alignment` |
| `analysis/risk_parity.ipynb` | 全流程 + 所有验证断言 + 图 |

---

## 2. 关键决策

| 项 | 决定 | 理由 |
|---|---|---|
| 模块依赖 | 单向：`risk_parity` → `backtesting_analysis` → `estimation` | `risk_contributions` 是通用工具（分析所有组合），放在分析模块；RP 只借用它检查 RC 均等 |
| 接入 backtest | 通过 `rolling_backtest(constructors=...)` 传入 | 不改 `backtest.py`，自动共享 universe / cov / delisting / turnover 口径 |
| RP 定义 | `RC_i = x_i (Vx)_i / sigma_p` 全部相等，long-only | 标准 ERC |
| 生产求解器 | single-factor 闭式：给定 `b = beta'y`，每个 `y_i` 是一元二次方程正根，bisection 解 `b` | 和 min risk 的 `beta_LO` 同构；0.6 ms vs cvxpy 1.3 s |
| 通用求解器 | Newton 解 `min 1/2 y'Vy - sum log y`（Spinu 2013） | deliverable 要求 `risk_parity_weights(Sigma)` 接受任意矩阵 |
| Metrics 口径 | **逐项对齐 `backtest.summarize`**：几何年化收益；vol / Sharpe 用 excess；drawdown 用复利路径；单边 turnover，对 drift 后权重算，首期 NaN | 两套代码可互相校验 |
| 子区间 turnover | 只计持有月落在该区间内的 rebalance | 子区间统计不串期 |
| 非月度 rebalance | `drifted_turnover` 对两次 rebalance 之间的所有月份复利 drift | 以防队友降频 |
| RC 诊断的 `V_t` | `estimate_covs` 按同一窗口、同一 universe 重估 | backtest 不存 cov；逐股回归独立，结果与 backtest 内完全相同 |

---

## 3. 验证（全部写成 notebook 里的 assert）

- 三种解法一致：closed form vs Newton 全 372 个月最大差 **6e-12**；vs cvxpy **1e-7**
- RC 均等：全 372 个月 `max |n p_i - 1|` = **2.9e-14**
- Euler 恒等式 `sum RC_i = sigma_p`：误差 < 1e-15
- 已知特例：对角 `V` → inverse-vol；等相关等波动 → EW；constant correlation → inverse-vol
- min risk 的 KKT：held names 的 MRC 全等于 `sigma_p`，**RC share = 权重**（4e-16）
- `performance_table` 与 `backtest.summarize` 五项指标差 < **1e-12**（turnover 是独立重算的）
- min risk 子区间数字与 COV_Notes 第 5 节完全一致

### 踩到的坑：Newton 收尾时 line search 卡死

接近最优时 Newton decrement ~ 1e-17，而 `f ~ 177`，浮点只能分辨 ~3e-14 的下降，
Armijo 永远不满足 → 步长被砍到 0，残差停在 6e-9。
**修正**：decrement < 1e-8 时跳过 line search 走 full Newton step（局部二次收敛）。

---

## 4. 主结果（1995-01 .. 2025-12，T=60，n=500，372 个 OOS 月）

|  | ann_ret | ann_vol | Sharpe | MDD | turnover | beta |
|---|---|---|---|---|---|---|
| Minimum risk | 8.20% | 12.72% | 0.499 | -37.6% | 116.6% | 0.34 |
| **Risk parity** | **11.62%** | **14.04%** | **0.689** | **-48.9%** | **62.4%** | **0.85** |

子区间：RP 在 1995-99 大胜（Sharpe 1.13 vs 0.31），2000-09 输（0.25 vs 0.37，GFC 回撤更深），
2010-19 打平，2020-25 胜。

### Risk concentration（372 月均值）

|  | n_held | eff_n_weight | eff_n_risk | top10 RC share | systematic share |
|---|---|---|---|---|---|
| Minimum risk | 45 | 25 | **25** | 54% | **73%** |
| Risk parity | 500 | 424 | **500** | 2% | **99%** |
| EW（参照） | 500 | 500 | 437 | 4.5% | 99% |

---

## 5. 最值得进 slides 的发现

1. **min risk 的风险集中度 = 权重集中度**（KKT：MRC 相等 ⇒ RC share = x_i）。
   effective N 25 既是持仓数也是风险源数。
2. **"每只股票风险相等" ≠ "风险来源分散"**。single-factor 下 RP 和 EW 99% 的 ex-ante 方差
   都来自市场因子；RP 只是在 500 个 idiosyncratic 部分之间平摊那 1%。
   只有 min risk 真正降低了市场暴露（systematic share 52%–88%，随 regime 变），
   这就是它 vol 更低、但牛市（1995-99）大输的原因。
3. **RP ≈ 往低 beta 倾斜的 EW**：EW 的 risk share 与 beta 几乎线性（notebook §3 右图），
   RP 把它压平；两者 eff_n_risk 437 vs 500，差距不大，turnover 也接近。

---

## 6. 待办

- [ ] 队友交来 `max_div` / `ew` / `vw` constructor 后，加进 notebook §4 的 `CONSTRUCTORS`，
      重跑；`check_alignment` 必须五个一起通过
- [ ] 五组合的 Exhibit 1 风格表直接由 `performance_table` 生成
- [ ] 最终打包只允许 1 notebook + 1 `.py`：`risk_parity.py` 与 `backtesting_analysis.py` 需随全组一起合并
- [ ] （可选）HRP 需要非 factor 的 `V`（见 COV_Notes 第 2 节），`risk_parity_weights` 已支持 dense 输入
