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
复用 `evaluation.py`，不另建一套 evaluation。

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
最终五策略比较已将 MaxDiv constructor 加入共同的 `constructors` dictionary，
再用共享 evaluation 和 alignment checks 生成比较表。

共同数据限制沿用其他部分：完整历史筛选偏向上市较久股票；missing next-month
return 按 RF 处理。美国普通股筛选在 `project_data.py` 下载 SQL 中完成，
分类字段未保留到缓存；已有缓存仍需保留来源记录。EW/VW 正式回测、
替代 covariance、position caps 和最终提交打包由项目其他部分统一处理。

共同 evaluation 已修正 running peak，将初始 wealth=1 纳入回撤计算。
首月 -1.19% 现被正确记录；全样本最大回撤仍为 -53.62%。

MaxDiv 已加入 `analysis/comparison.ipynb` 的五策略共同回测、绩效表、
累计收益、回撤、子区间以及 DR / risk concentration 比较。

## 8. 独立复核（2026-10-02）

**结论：在当前共同 covariance、universe 和缺失收益规则下，未发现 MaxDiv 推导、求解或收益计算错误；核心研究内容符合 project.pdf Topic 3。**
这不代表真实未来表现有保证，也不代表数据简化假设消失。

### 数学核查

DR 最大化等价于 `min y'Vy`，约束 `sigma'y=1, y>=0`；
`y=x/(sigma'x)`，归一化恢复 `x=y/sum(y)`。
令 `z=Diag(sigma)y`，相关矩阵 `C=rho rho'+Diag(d)`，则
`min z'Cz`、`sum(z)=1`。KKT 条件给出持有股票上的
`d_i*z_i + rho_i*(rho'z) = q`，其中 `q=z'Cz`。
正 beta 情形设 `h=q/(rho'z)`，由 `rho'z` 的一致性得到
`sum rho_i*(h-rho_i)^+/d_i=1`。因此实现的阈值方程与权重公式一致。
非正 beta 使用通用 QP，不套用正 beta 的阈值方程；零 market variance
的解为 inverse volatility，而不是 inverse variance。

### 本次实际运行的独立验证

- 15 个 synthetic dense QP 对照：1/2/10/50/100 assets，各含 positive-beta、
  mixed-beta、diagonal 情形；最大 DR 差 **1.08e-11**。
- 9 个直接以 DR 为目标的 SLSQP 对照（没有 QP 变量变换）：最大 DR 差
  **4.09e-14**；30 个 covariance scaling 检查通过。
- 8 个真实 500-asset dense QP 对照，涵盖首末期、不同年代和 mixed-beta
  月份；最大 DR 差 **2.04e-10**。参考 QP 显式构建 dense V，使用原始
  `sigma'y=1` 约束，而不是复用实现的 factor-form QP。
- 372 个月均独立重建 top-500 universe，确认只使用 t 及之前的 60 个月
  数据，权重在 t 设置并获得 t+1 收益；long-only、fully-invested、
  EW/VW 对齐检查通过。
- 用 centered OLS 独立重估所有月份的 covariance；与共享 normal-equation
  实现的 beta / residual variance 最大绝对差 **3.55e-15**。
- 全样本独立 KKT 证书（correlation-coordinate、相对 q 归一化）：最大
  residual **7.67e-9**。与 §4 的 `2.85e-9` 使用不同坐标/归一化口径。
- 338 个正 beta 月份用排序和累计求和求阈值，完全不使用 bisection；
  最大权重差 **4.77e-14**。其余 **34** 个月使用 QP。
- 全部 372 个月、MaxDiv/EW/VW 的收益和 drifted turnover 独立重算：
  最大绝对差分别 **5.55e-17**、**1.54e-16**。
- 用独立公式重算几何年化收益、excess-return vol / Sharpe 和包含初始本金
  的 max drawdown；三个策略的最大指标差 **2.22e-16**。
- MaxDiv 在每个 formation date 的 DR 均不低于 EW 和 VW；平均 DR
  为 **3.660748 / 2.116663 / 1.945934**（MaxDiv / EW / VW）。

### Return 低于 EW/VW 的解释边界

MaxDiv 优化的是估计 covariance 下的 DR，不使用 expected-return forecast，
因此不存在必须跑赢 EW/VW 的收益约束。只有在期望 excess return 向量与
`sigma` 成比例等附加假设下，DR 才与模型 Sharpe 成比例；本项目没有估计或
验证这一收益假设。即使成立，也不能保证 realized return 排名。

本样本 MaxDiv 年化 return **10.35%**，EW **11.63%**，VW **11.55%**；
vol 分别 **14.11% / 15.74% / 14.82%**，Sharpe 分别
**0.604 / 0.631 / 0.656**。MaxDiv 实际收益较低的具体原因没有被这些核查
识别，不能直接归因于某个 sector、缺失收益或 covariance error。

### 要求与限制

Topic 3 要求 fully-invested long-only 三种优化策略，与 EW/VW 的共同
out-of-sample summary 和 cumulative-return comparison，没有要求任何
优化策略必须跑赢 benchmark。MaxDiv 核心内容及五策略比较已满足这些项。
匿名 slides、可复现 demo 打包和课程参考 Exhibit 的版式复核属于全组交付。

共同限制仍需披露：完整历史筛选偏向较老股票；missing next-month return
按 RF 计是简化假设；single-factor 忽略 residual cross-correlation；结果未扣
交易成本。下载 SQL 中有 common-stock 筛选，但缓存本身未保留分类字段，
因此本次无法仅凭缓存独立确认下载来源，也未用 WRDS 重新下载数据。
HW4 原题及教授代码不在本 repository；本次验证的是项目描述与数学等价性，
并非对缺失原文件作逐字逐行核对。
