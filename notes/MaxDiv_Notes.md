# Maximum Diversification —— 方法与结果记录

> 本部分负责 topic 3（Comparison of Diversification Approaches）的 **maximum
> diversification portfolio**。与 minimum risk、risk parity 共用 covariance
> estimate、universe、rolling backtest 和 performance evaluation。

## 1. 目标与核心设计

最大化 fully-invested、long-only portfolio 的 diversification ratio：

`DR(x) = sigma'x / sqrt(x'Vx)`，其中 `sigma = sqrt(diag(V))`。

采用共同的 single-factor covariance：
`V = sigmaM2 * beta beta' + Diag(omega2)`。
`sigmaM2` 与 `omega2` 都是 variance；分子中的
`sigma` 使用模型值，使目标函数和 HW4 公式保持一致。

与 minimum risk、risk parity 相同，monthly backtest 使用 single-factor
解析结构和 bisection，CVXPY 模型用于 numerical cross-validation。

## 2. 文件与接口

| 文件 | 内容 |
|---|---|
| `max_diversification.py` | QP、threshold bisection、DR、KKT diagnostics |
| `analysis/max_diversification.ipynb` | 数据、方法、单次调仓验证、完整回测、绩效与分散化诊断 |

组合构造函数接收共同的 `FactorCov`，返回 `x`、`sigma2`、`n_held`；
闭式解另返回 `rho_LO`。通过 `rolling_backtest(constructors=...)` 接入，
`**_` 接收共享 loop 传入的 `caps` 等参数。

DR 和 KKT 在验证、诊断阶段计算。绩效、turnover、risk concentration
复用 `performance.py`，不另建一套 evaluation。

## 3. 数学方法与关键决策

### QP：HW4 Exercise 2(b)

令 `y = x/(sigma'x)`，求解
`min y'Vy` subject to `sigma'y=1, y>=0`，最后 `x=y/sum(y)`。

代码进一步使用 `z_i=sigma_i*y_i`，约束成为 `sum(z)=1`，目标写成
factor form。该变量缩放与教授解答的 dense QP 数学等价。

### Threshold：HW4 equation (4) / Exercise 2(d)

正 beta、正 market variance 下，定义
`rho_i=sqrt(sigmaM2)*beta_i/sigma_i`、`d_i=omega2_i/sigma_i^2`。
Bisection 解标量方程：

`sum rho_i * max(h-rho_i, 0) / d_i = 1`。

随后计算 `u_i=sigma_i/omega2_i * max(1-rho_i/h, 0)`，
`x=u/sum(u)`，`rho_LO=h`。仅持有 `rho_i<rho_LO` 的股票。

| 项 | 决定 | 理由 |
|---|---|---|
| 回测求解 | Threshold bisection | 保留 factor structure，与另两种策略一致 |
| QP solver | Clarabel → SCS | 无需 commercial license |
| 非正 beta | 使用通用 QP | 正 beta 的阈值推导不覆盖这种情形 |
| 零 market variance | Inverse-volatility weights | 对角 covariance 的 MaxDiv 解 |
| QP 数值残余 | 小于 `1e-8` 的权重截零并归一化 | 与 min risk 同样处理 numerical dust，避免虚增持仓数 |
| 数据与交易口径 | 使用共享 loop | 相同 universe、估计窗口、missing-return 和 turnover 规则 |

## 4. 验证

Notebook 中保留真实首期 QP 对照、公式 (4) 重建、全样本 KKT 和绩效检查。

- 教授的 10-asset case 和随机 10/50/100-asset case：与独立 dense QP 的 DR 一致。
- 真实首期 rebalance：QP 与 closed form 的 DR 差小于 `1e-6`；公式 (4) 重建权重通过。
- 全样本 relative KKT residual 最大约 **2.85e-9**；每个月 DR 不低于同 universe 的 EW。
- 对角 covariance、单资产、covariance scaling、mixed-beta QP 和求解失败情形通过。
- `check_alignment` 通过；五项共同绩效指标与 `backtest.summarize` 差小于 **1e-12**。
- 本次独立复核的 7 个 dense QP 对照中，最大 DR 差约 **4.37e-11**。
- 372 个月每月均为 500 股票 universe；其中 **34 个月**因非正 beta 使用 QP。
  从保存权重和次月收益逐月重算，最大收益差约 **2.78e-17**。

## 5. 主结果

1995-01 .. 2025-12，`T=60`、`N=500`，monthly rebalance，372 个 OOS 月。
绩效口径与 RP 相同：geometric annual return、excess-return vol / Sharpe、
compounded drawdown、drifted one-way turnover；未扣 transaction costs。

| Strategy | Annual return | Annual vol | Sharpe | Max drawdown | Annual turnover | Realized beta |
|---|---|---|---|---|---|---|
| Maximum diversification | 10.35% | 14.11% | 0.604 | -53.62% | 160.36% | 0.673 |

### 子区间

| Period | Annual return | Annual vol | Sharpe | Max drawdown | Realized beta |
|---|---|---|---|---|---|
| 1995–1999 | 13.91% | 16.39% | 0.576 | -25.18% | 0.951 |
| 2000–2009 | 4.39% | 13.53% | 0.185 | -53.62% | 0.514 |
| 2010–2019 | 15.89% | 12.52% | 1.207 | -15.39% | 0.748 |
| 2020–2025 | 8.62% | 15.36% | 0.442 | -21.44% | 0.707 |

### Diversification / risk concentration（monthly time averages）

EW 仅用于共同 covariance 上的诊断参照，不在此处单独回测。

| Portfolio | DR | Holdings | Effective weight N | Effective RC count | Top 10 RC share | Market variance share |
|---|---|---|---|---|---|---|
| Maximum diversification | 3.661 | 63.0 | 36.7 | 38.0 | 41.13% | 65.48% |
| EW reference | 2.117 | 500.0 | 500.0 | 437.1 | 4.53% | 99.12% |

## 6. 主要发现

1. **MaxDiv 最大化 DR，不等于最大化持仓数。** 平均只持有约 63/500 只股票，
   effective weight N 约 37，但模型中的 DR 明显高于 EW。
2. **Stock concentration 与 factor concentration 是不同问题。** MaxDiv 的
   market variance share 约 65%，低于 EW 的 99%；effective RC count 衡量的是
   股票 RC 集中度，不能直接当作 independent risk sources 数量。
3. **Ex-ante 最优不保证 realized performance 最优。** 2000–2009 的 Sharpe
   仅 0.185，2010–2019 为 1.207；turnover 也较高。优化结果与 covariance
   forecasting quality、交易成本需要分开评价。

## 7. 与项目其他部分的衔接

本 notebook 聚焦 MaxDiv，与 min risk、RP 作为平行的策略部分。
最终五策略比较将 MaxDiv constructor 加入共同的 `constructors` dictionary，
再用共享 evaluation 和 alignment checks 生成比较表。

共同数据限制沿用其他部分：完整历史筛选偏向上市较久股票；当前 extract
缺 share type；missing next-month return 按 RF 处理。EW/VW 正式回测、
替代 covariance、position caps 和最终提交打包由项目其他部分统一处理。

共同 evaluation 尚有一个具体问题：`performance.drawdowns`、
`performance_metrics` 和 `backtest.summarize` 的 running peak 未包含初始 wealth=1。
首月亏损时，回撤曲线会低估起始回撤；本样本首月为 -1.19%。
纳入初始 wealth 后，全样本最大回撤仍为 -53.62%，但曲线起始部分应由共同评估层统一修正。
