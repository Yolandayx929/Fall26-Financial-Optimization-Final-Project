# Covariance Estimation 模块 —— 进度与决策记录

> 本人负责 topic 3（Comparison of Diversification Approaches）的 **covariance estimation**
> 一段。数据集与 preliminary 由队友负责，本文件只记录本模块。
> 统计、金融与一般技术术语用 English，其余用中文。

---

## 1. 目标

Topic 3 要在同一个 universe、同一套 covariance 估计下，比较
min risk / max diversification / risk parity 与两个 ad-hoc 组合（VW、EW）的
out-of-sample 表现。本模块产出下游优化器消费的风险估计。

**本模块最重要的设计决定不是用哪种 shrinkage，而是接口形态。**
在 single-factor 下，三个优化器真正需要的是 `(beta, omega2, sigmaM2)` 三元组，
HW4 Q2 的式 (3)(4)(5) 据此可以用 bisection 直接求解；HW4 2(d) 原文说这比优化
*"substantially more efficient"*。若压成 500×500 的 dense `V` 交出去，closed form 就用不了，
而 demo 必须让 peer reviewer 用自己的 WRDS 账号跑通（关联 5 分 peer review）。

---

## 2. 当前已完成

本轮只做**最基础版本**：矩阵 OLS + 单一 shrinkage，主要沿用 homework 的做法。

| 文件 | 内容 |
|---|---|
| `estimation.py` | `FactorCov` dataclass + `estimate_single_factor`（HW3 Q2(c) 做法） |
| `portfolios.py` | `min_risk`（cvxpy，HW4 Q2(a)）、`min_risk_closed_form`（式 (3)，验证用）、`equal_weight`、`value_weight` |
| `backtest.py` | `select_universe`、`rolling_backtest`、`summarize` |

### `FactorCov` 接口

```
permno, beta, omega2, sigmaM2, sigma, beta_hat, alpha
.quad(x)         -> x'Vx，走 factor 形式，不构造 dense V
.exante_vol(x)   -> sqrt(x'Vx)
.exante_beta(x)  -> beta'x
.to_dense()      -> 仅测试/验证用
```

### 已验证（synthetic data）

- `sigmaM`、`mean(beta)` 量级正确；shrunk beta 的 SSE < OLS beta 的 SSE
- `cov.quad(x) == x @ cov.to_dense() @ x`；`diag(V) == sigma**2`
- **`min_risk`（cvxpy）与 `min_risk_closed_form`（式 3）权重对齐到 3.6e-5，
  variance 对到 10 位，`n_held` 一致** —— 即 HW4 Q2(a) 要求的数值验证
- 端到端 rolling backtest 跑通；EW return 等于 universe 内简单平均；
  权重和为 1、非负；min risk 的 realized vol 低于 EW

---

## 3. 关键决策点

### 3.1 已定

| 项 | 决定 | 理由 |
|---|---|---|
| Covariance | Single-factor，`V = sigmaM2 * beta beta' + Diag(omega2)` | project.pdf 对 topic 3 明文建议；且 HW4 的 closed form 可用 |
| 回归形式 | market model，**保留截距**，两边都是 excess return | 不是 CAPM；关心的是 covariance 结构，不强制 alpha = 0 |
| `omega2` 自由度 | `T - 2` | 估了 alpha 和 beta |
| Shrinkage | **只做 beta，Blume `2/3*beta_hat + 1/3`** | 对齐 HW3 Q2(c)；本轮只保留一种 shrinkage |
| `sigmaM2` | 同一 T 窗口，`ddof=1` | 与 `se(beta_hat)` 用同一 `rM`，内部一致 |
| Max div 的 `sigma` | **必须取 `sqrt(diag(V))`（模型值）**，不得另用 sample std | 否则 `sigma` 与 `V` 不一致，式 (4) 的闭式解失效 |
| Solver | 不用 GUROBI；Clarabel → SCS → ECOS 依次回退 | peer reviewer 可能没有 license |
| 目标函数写法 | factor 形式 `sigmaM2*(beta'x)^2 + sum omega2_i x_i^2` | 更快，且天然 PSD，避免 dense V 的数值问题 |
| 缺失值 | `estimate_single_factor` 收到 NaN 直接报错 | universe 选择是上游的职责，不做静默填补 |

### 3.2 实现中遇到、已解决

**cvxpy 的 interior-point solver 不产生精确的零。**
被排除的股票会拿到 ~1e-6 量级的 dust，导致 `n_held` 从 40 虚报成 68。
实测 dust 最大 1.7e-6，而真实最小持仓 4.2e-4，相差两个数量级。
处理：`min_risk` 增加 `weight_tol=1e-5`，低于阈值截零后重新归一化。
`min_risk_closed_form` 才是"哪些股票真被持有"的权威。

**Turnover 必须对 drift 后的权重算。**
`backtest.py` 里保存的 `previous` 是 `x * (1+r) / (1+r_p)`，不是原始 `x`，
否则会把被动漂移算成主动交易。第一期的 turnover 记为 `NaN`，排除在均值之外。

**Delisting 的 out-of-sample 口径。**
持仓在 `t+1` 无 return 时，该权重按 `rf` 计。
不按存活者归一化 —— 那等于月初就知道谁会消失（look-ahead）。

### 3.3 待定（不阻塞当前代码）

- [ ] `crsp.msf_v2` 的 share type / security type 字段名；`mthret` 是否已含 delisting return
- [ ] 是否加 residual volatility 的 log-vol shrinkage（本轮未做）
- [ ] 是否加 cross-sectional winsorize。**注意**：截断原始 return 会在市场大幅波动的月份
      系统性削掉 high beta 股票（截面离散度随 |market move| 放大，高 beta 落在尾部），
      与 Blume shrinkage 叠加成双重收缩。若要做，建议截断 **residual** 而非原始 return
- [ ] 是否加 Vasicek beta shrinkage 作为对比
- [ ] 是否加个股上限 3%/5%（注意：会破坏 closed form，只能走 cvxpy）
- [ ] 与 person 3 敲定 `FactorCov` 字段名，以及 bisection 求解器归属

---

## 4. 从 project.pdf 读出来、影响范围判断的几条

逐句量过 topic 3 之后：

- **强制交付面很小**：五个组合 + 一张 Exhibit 1 风格的表 + 一张 Exhibit 4 风格的图。
- **turnover 和个股上限都是 topic 1 的句子**，topic 3 从头到尾没提 —— 是自选项不是要求。
- **covariance zoo 是 topic 1 的得分面**。topic 1 的 Goal 句就是
  *"via various approaches to covariance estimation"*；topic 3 的 Goal 句是
  diversification 比较，对 covariance 只说 *"feel free to use other"* —— 是许可，不是奖励。
  所以本模块实现几个、slides 上报几个，应该分开决定。
- **PDF 自己点名的唯一 robustness 轴是 test periods**
  （*"different test periods that span different kinds of economic conditions"*），
  不是 covariance estimator，也不是 n。
- **HW4 式 (3)(4)(5) 就是 CdST 2013 的解析解**（notation `sigma2_LMV` / `beta_LO` /
  `rho_LO` / `sigma_LA` / `gamma` 一致）。instructor 指定那篇当表格范本的用意，
  应该是让我们把解析结构和 out-of-sample 结果连起来：
  min risk 只持 `beta_i < beta_LO` 是 low-beta bet；
  max div 只持 `rho_i < rho_LO` 是 low-correlation bet；risk parity 全持有。
- 教材 `OIFbook2026.pdf` §7.5 原文给了统一视角：max div 的 diversification ratio
  *"is proportional to the Sharpe ratio if mu is proportional to sigma"*，
  即各组合都是某套 implicit return assumption 下的 max-Sharpe。

---

## 5. 下一步

1. 接真实 CRSP 数据，核对单位：月度 `sigmaM` 应在 **0.04–0.05**；
   若是 4–5，说明 `MktRf` 忘了 ÷100
2. 跑小规模（n=100、近 10 年）确认全流程，再跑全样本
3. 和 person 3 对接 `FactorCov`，补 `max_div` 和 `risk_parity`
4. 视队伍决定，再加 shrinkage 的 ablation（不 shrink → 只 shrink beta → beta + omega）

> **注**：当前 synthetic 测试里的 `ann_turnover` 偏高（EW 4.3/年），
> 是因为测试数据的 market cap 每月独立重抽，universe 每月大换血。
> 真实数据上 cap 高度持续，EW/VW 的 turnover 会小得多。

---

# 追加：真实数据跑通（`min_risk_port.ipynb`）

## 做了什么

`data/` 到位后（`crsp_msf_v2.parquet` + `MktRf.csv`），建了 `min_risk_port.ipynb`，
逐条对应 project.pdf 第 2 页 (a)–(d)。**全样本 372 个 OOS 月份，backtest 1.7 秒跑完，无报错。**

模块侧的三处改动：

1. `portfolios.py` 从 `optimal_ports/` 移回根目录 —— 之前 `backtest.py` 是 `ModuleNotFoundError`
2. `min_risk_closed_form` 加 `**_`，这样能直接进 `CONSTRUCTORS`
3. `backtest.py` 的 `CONSTRUCTORS["min_risk"]` **默认指向闭式解**，并记录 `beta_LO`

## 主结果（1995-01 .. 2025-12，T=60，n=500）

| | ann_return | ann_vol | Sharpe | max_DD | **pred/realized** | avg_n_held | ann_turnover |
|---|---|---|---|---|---|---|---|
| Minimum risk | 0.0820 | **0.1272** | 0.4992 | -0.3755 | **0.5741** | 45 | 1.17 |
| Value weighted | 0.1155 | 0.1482 | **0.6557** | -0.4913 | 1.0015 | 500 | 0.10 |
| Equally weighted | 0.1163 | 0.1574 | 0.6313 | -0.5170 | 0.9682 | 500 | 0.62 |

- min risk 把 realized vol 降到 12.72%，比 VW 低 14.2% —— **它该做的事做到了**
- 但 **Sharpe 没赢**（0.50 vs 0.66 / 0.63），降波动付了收益代价
- avg beta **0.43**、平均只持 **45 / 500** 只 —— 闭式解说的 low-beta bet，实证完全对上

## 最重要的发现：模型严重低估 min risk 的风险

`predicted / realized vol`：**min risk 0.574**，EW 0.968，VW 1.002。

模型只预测到 min risk 实际波动的 **57%**，但对两个宽基组合几乎无偏。
**这个不对称排除了 scaling / annualization bug**（那种错会三个一起错）。
原因是 single-factor 的 **uncorrelated residual 假设**在最该失效的地方失效了：
min risk 集中在 45 只低 beta 股票上，这些股票扎堆在少数防御性行业，residual 正相关。

**这是全套结果里最强的一页**，也是"要么换更丰富的 covariance，要么加个股上限"的直接论据。

## 子区间：验证了子区间切分的那个判断

| Episode | min risk | VW | 差 |
|---|---|---|---|
| dot-com bust 2000-02（31m） | **+42.1%** | -34.8% | **+76.9pp** |
| GFC 2007-09（16m） | -37.1% | -49.1% | +12.0pp |
| COVID 2020（2m） | -15.6% | -18.9% | +3.3pp |
| 2022（12m） | +4.3% | -18.3% | +22.6pp |

对照 1995-1999：min risk 年化 +8.5%，VW +28.9% —— **min risk 大输**。

所以之前说的"1995–2007 一个桶会把 dot-com 的大胜和 90 年代末的大输抵消掉"是对的，
数据证实了。子区间必须把 2000-02 单独拿出来。

> 注：episode 表用**累计收益**不用年化。COVID 只有 2 个月，年化会得到 -80%、Sharpe -14.5
> 这种看着吓人但没有信息量的数字。

## 数据层面踩到的坑

1. **`mthcaldt` 是交易日月末，不是日历月末**（1994-12-31 是周六 → 记录是 1994-12-30）。
   所以 index 和 join **一律用 `yyyymm` 整数**。MktRf.csv 的 index 正好也是 `yyyymm`，对得上。
2. **`MktRf.csv` 是百分点，必须 ÷100**。验证：60 月窗口的 `sigma_M` = 0.0439，在 0.04–0.05 内。
3. **这份 extract 没有 `sharetype` / `securitytype` 列**，做不了标准 CIZ 筛选。
   只能 `primaryexch ∈ {N, A, Q}`（丢掉 1.80% 的行）。notebook 里已注明这个限制。
4. `mthret` 是小数，`mcap = mthcap / 1000`（VW 用哪个都一样，scale-invariant）。

## cvxpy vs 闭式解：实测

| | 单次耗时 | variance | n_held |
|---|---|---|---|
| 闭式解（式 3） | **0.23 ms** | 0.00046475 | 42 |
| cvxpy | 670 ms | 0.00046476 | 45 |

**闭式解的目标值严格更低**（cvxpy 差 1.7e-6 相对）。差异来自 3 只 beta 落在
0.6458–0.6504、刚好**高于** `beta_LO = 0.6448` 的股票：它们本就该被排除，闭式解排对了，
cvxpy 的 interior-point 在边界上留了 1.7e-4 量级的 dust。

**所以验证要比目标值，不是逐元素比权重。** notebook 里的断言已经改成这样。

## 下一步

- person 3 接 max div / risk parity（接口已验证可用，无需改动）
- 打包约束：project.pdf 只允许「1 个 notebook + 最多 1 个 .py + 1 个 .csv」，
  现在是 3 个 .py，**最终交付前要合并**
