# Covariance Estimation + Minimum Risk —— 进度与决策记录

> 本人负责 topic 3 的 **covariance estimation**（single-factor 模型、beta shrinkage）
> 和 **minimum risk** 组合，以及 backtest 的滚动框架。
> 不改动队友的 `risk_parity.py` / `evaluation.py` / `benchmarks.py` / `project_data.py`。
> EW / VW 两个 benchmark 不归本部分。

---

## 1. 新增文件

| 文件 | 内容 |
|---|---|
| `estimation.py` | `FactorCov` dataclass + `estimate_single_factor`（HW3 Q2(c) 的矩阵回归做法） |
| `min_risk.py` | `min_risk`（cvxpy，HW4 Q2(a)）、`min_risk_closed_form`（式 (3)，生产用） |
| `backtest.py` | `select_universe`、`rolling_backtest`、`summarize`（全组共用的滚动框架） |
| `analysis/min_risk_port.ipynb` | 全流程，逐节对应 project.pdf 第 2 页 (a)–(d) |

---

## 2. Covariance 是怎么估的

用 **single-factor model**（市场模型），不是 CAPM —— 我们关心的是协方差结构，不是定价。
协方差矩阵是 `V = sigmaM2 * beta beta' + Diag(omega2)`。

- **一次矩阵回归估完 500 只股票**（同 HW3）：`B = solve(X'X, X'R)`，`X = [1, rM]`，两边都是 excess return，**保留截距**。
- **残差方差 `omega2` 用 `T-2` 自由度**，因为 alpha 和 beta 都估了。
- **beta 做 Blume shrinkage**：`beta~ = 2/3 * beta_hat + 1/3`。
  *大白话：回归出来的 beta 噪音很大，真实 beta 有向 1 回归的倾向，所以把它往 1 拉三分之一。*
  实测截面标准差从 0.4837 压到 0.3224。
- **只做这一种 shrinkage**，residual vol 不 shrink（见第 7 节待办）。

**为什么不交 dense V：** 三个优化器真正要的只是 `(beta, omega2, sigmaM2)` 三元组。
留着这个结构，HW4 Q2 的闭式解就能用；压成 500×500 的矩阵就用不了了。
**实测 n=500 时闭式解 0.25 ms、cvxpy 746 ms（3000 倍）**，全样本 backtest 因此只要 1.5 秒。
demo 要让 peer reviewer 自己跑通（关联 5 分），这个差距直接决定交付物能不能用。

---

## 3. `FactorCov` 接口（队友要用的）

```
permno, beta, omega2, sigmaM2, sigma, beta_hat, alpha
.quad(x)         -> x'Vx，走 factor 形式，不构造 dense V
.exante_vol(x)   -> sqrt(x'Vx)
.exante_beta(x)  -> beta'x
.to_dense()      -> 仅测试/验证用
```

**已验证能直接支撑 max div 和 risk parity**：照这个接口两个都写出来跑过，
cvxpy 与闭式解一致（max div 权重差 9.5e-6，`rho_LO` = 0.405；
risk parity 的 risk contribution 相对离散 9.9e-5）。**无需改动。**

**已知限制：HRP 用不了这个 V。** single-factor 下 correlation 的非对角部分精确是 rank-1
（`rho_ij = rho_i * rho_j`，实测 `max |C_ij - rho_i*rho_j| = 1.67e-16`），
clustering 没有 block 结构可发现。要做 HRP 得另给非 factor 的 `V`。

---

## 4. 关键决策

| 项 | 决定 | 大白话 / 理由 |
|---|---|---|
| Covariance | single-factor | project.pdf 对 topic 3 明文建议，且闭式解可用 |
| 回归形式 | 保留截距，两边 excess return | 是统计模型不是 CAPM，不强制 alpha = 0 |
| beta shrinkage | Blume `2/3 + 1/3` | 对齐 HW3 Q2(c)，只留一种 |
| Max div 的 `sigma` | 必须取 `sqrt(diag(V))` | 另用 sample std 会和 `V` 不一致，式 (4) 失效 |
| 生产求解器 | 闭式解；cvxpy 只做交叉验证 | 3000 倍差距 |
| Solver | 不用 GUROBI，Clarabel → SCS → ECOS | peer reviewer 可能没 license |
| Rebalance | 每月 | project.pdf 允许降频但理由限定为 *"time constraints"*，闭式解让该前提不成立 |
| 缺失值 | 收到 NaN 直接报错 | universe 选择是上游职责，不静默填补 |
| Delisting | `t+1` 无 return 的权重按 `rf` 计 | 按存活者归一化 = 月初就知道谁会消失（look-ahead） |
| Turnover | 对 **drift 后**的权重算，首期 NaN | 否则把被动漂移算成主动交易 |

**三个容易踩的坑（都已处理）：**

- **cvxpy 不产生精确的零。** 被排除的股票会留 1e-6 量级的 dust，`n_held` 严重虚报。
  真实数据上闭式解 42 只、cvxpy 45 只，差的 3 只 beta 是 0.6458–0.6504，**刚好高于**
  `beta_LO = 0.6448`，本就该排除；闭式解目标值**严格更低**。
  *所以验证要比目标值，不是逐元素比权重。*
- **`mthcaldt` 是交易日月末不是日历月末**（1994-12-31 是周六 → 记录是 1994-12-30）。
  一律用 `yyyymm` 整数做 index 和 join。
- **`MktRf.csv` 是百分点，必须 ÷100。** 验证：60 月窗口 `sigma_M = 0.0439`，在 0.04–0.05 内。

**更正一条我之前写错的：** 我曾说"extract 没有 share type 列，做不了 CIZ 筛选，是已知限制"。
**不对** —— 看 `project_data.py` 才知道筛选是在下载的 SQL WHERE 里做的，
parquet 里已经只有美国普通股。**这条不用写进 slides。**

---

## 5. 主结果（min risk，1995-01 .. 2025-12，T = 60，n = 500，372 个 OOS 月）

| ann_return | ann_vol | Sharpe | max_drawdown | ann_turnover | ex-ante beta |
|---|---|---|---|---|---|
| 8.20% | **12.72%** | 0.499 | **−37.6%** | 117% | 0.427 |

闭式解说的 **low-beta bet，实证完全对上**：平均只持 45/500 只，effective N 25，
`beta_LO` 均值 0.578（范围 0.437–0.707）。

**注意别和队友表里的 0.34 搞混** —— 那是 **realized** beta（对 Mkt-RF 回归），
我这里是 **ex-ante** beta（`beta'x`）。两个是不同的量，都对。

### 子区间

| | ann_ret | ann_vol | Sharpe | avg_n_held | avg_beta |
|---|---|---|---|---|---|
| 1995-1999 | 0.0848 | 0.1308 | 0.307 | 51 | 0.489 |
| 2000-2009 | 0.0702 | 0.1340 | 0.373 | 57 | 0.336 |
| 2010-2019 | 0.1275 | 0.1052 | **1.149** | 35 | 0.456 |
| 2020-2025 | 0.0267 | 0.1448 | 0.069 | 36 | 0.480 |

Episode 累计收益：dot-com bust 2000-02 **+42.1%**、GFC 2007-09 −37.1%、
COVID 2020 −15.6%、2022 **+4.3%**。

**子区间必须把 2000-02 单独切出来。** 同期 VW 是 −34.8%（差 76.9pp），
而 1995-99 期间 min risk 年化 +8.5%、VW +28.9%，大输。
一个 1995-2007 的桶会把两者抵消掉，什么也看不出来。

> Episode 用**累计**不用年化：COVID 只有 2 个月，年化会得到 −80%、Sharpe −14.5
> 这种看着吓人但没信息量的数字。

---

## 6. 最值得进 slides 的发现

**模型只预测到 min risk 实际波动的 57%**（`predicted / realized vol` = 0.5741）。

这个数单看没有意义 —— 完全可能是 scaling 或 annualization 错了。
所以拿**随机 long-only 组合**做对照，同一个 universe、同一个 covariance model 定价：

| | mean ratio | range | 大白话 |
|---|---|---|---|
| **Minimum risk** | **0.5741** | — | 模型严重低估它的风险 |
| Random, diffuse（500 只随机权重） | 0.9688 | 0.9650–0.9732 | 模型对普通组合很准 |
| **Random, concentrated（随机 45 只）** | **0.9740** | 0.9489–0.9932 | **一样集中，但模型照样很准** |

**第二行对照是关键：集中度本身不是原因。** 随机抽 45 只、和 min risk 一样集中，
模型定价准到 97%。所以问题不在"持股少"，而在**优化器挑的是哪些股票** ——
它系统性选中 residual 正相关的那批低 beta 防御股，而 single-factor 假设 residual 不相关。

**还有一个独立佐证：** 模型**高估**市场暴露（ex-ante beta 0.43 vs realized 0.34），
却**低估**总波动。既然实际市场暴露比模型以为的还小，多出来的实际波动就只能来自非市场来源。

这是"要么换更丰富的 covariance，要么加个股上限"的直接论据。

---

## 7. 待办

- [ ] 是否加 residual volatility 的 log-vol shrinkage（目前没做）
- [ ] 是否加 Vasicek beta shrinkage 做对比
- [ ] 是否加 cross-sectional winsorize。**注意**：截断原始 return 会在市场大幅波动的月份
      系统性削掉 high beta 股票，与 Blume 叠加成双重收缩；要做建议截断 **residual**
- [ ] 是否加个股上限 3%/5%（会破坏闭式解，只能走 cvxpy）。第 6 节的发现支持加
- [ ] shrinkage ablation（不 shrink → 只 shrink beta → beta + omega）
- [ ] **打包：** 只允许 1 notebook + 1 `.py`，现在全组 7 个 `.py` + 3 个 notebook，要一起合并

---

## 8. 怎么用这个模块

**只要一个 `FactorCov`：**

```python
from estimation import estimate_single_factor
cov = estimate_single_factor(R, rM, permno=universe.values)   # R: T x n excess returns
```

**接进 backtest：** 写一个 constructor，输入 `cov`（可选 `caps`），
返回 `dict(x=权重, sigma2=预测方差, n_held=持有数)`，然后放进
`rolling_backtest(constructors=...)`。这样自动和全组共享同一个 universe、
covariance、delisting 和 turnover 口径。

**跨组合比较提醒：** min risk vs EW vs VW 的对比在 `analysis/comparison.ipynb`。
谁加新策略都必须保证三点口径一致，否则表没法并排放：
universe 规则（top-500 by cap + complete 60m + `primaryexch ∈ {N,A,Q}`）、
`rf`（MktRf.csv ÷100）、delisting 口径（`t+1` 无 return 按 `rf` 计）。
