# Covariance Estimation + Minimum Risk —— 进度与决策记录

> 本人负责 topic 3（Comparison of Diversification Approaches）的 **covariance estimation**
> 和 **minimum risk portfolio** 两段。数据集由队友提供；EW / VW 两个 benchmark 不归本部分。
> 统计、金融与一般技术术语用 English，其余用中文。
>
> 最后更新：全样本跑通，范围已收窄到 min risk only。

---

## 1. 目标与核心设计决定

Topic 3 要在同一个 universe、同一套 covariance 估计下，比较
min risk / max diversification / risk parity 与两个 ad-hoc 组合（VW、EW）的
out-of-sample 表现。本模块产出下游优化器消费的风险估计，外加 min risk 组合本身。

**本模块最重要的设计决定不是用哪种 shrinkage，而是接口形态。**
在 single-factor 下，三个优化器真正需要的是 `(beta, omega2, sigmaM2)` 三元组，
HW4 Q2 的式 (3)(4)(5) 据此可以用 bisection 直接求解；HW4 2(d) 原文说这比优化
*"substantially more efficient"*。若压成 500×500 的 dense `V` 交出去，closed form 就用不了。

**实测证实了这个判断**：n=500 时闭式解 0.25 ms、cvxpy 746 ms（**3000 倍**），
全样本 backtest 因此只要 1.5 秒。而 demo 必须让 peer reviewer 用自己的 WRDS 账号跑通
（关联 5 分 peer review），这个差距直接决定交付物能不能用。

---

## 2. 当前文件

| 文件 | 内容 |
|---|---|
| `estimation.py` | `FactorCov` dataclass + `estimate_single_factor`（HW3 Q2(c) 做法） |
| `portfolios.py` | `min_risk`（cvxpy，HW4 Q2(a)）、`min_risk_closed_form`（式 (3)，生产用） |
| `backtest.py` | `select_universe`、`rolling_backtest`、`summarize` |
| analysis/`min_risk_port.ipynb` | 全流程，逐节对应 project.pdf 第 2 页 (a)–(d) |

### `FactorCov` 接口

```
permno, beta, omega2, sigmaM2, sigma, beta_hat, alpha
.quad(x)         -> x'Vx，走 factor 形式，不构造 dense V
.exante_vol(x)   -> sqrt(x'Vx)
.exante_beta(x)  -> beta'x
.to_dense()      -> 仅测试/验证用
```

**已验证可直接支撑 person 3**：照这个接口把 max div 和 risk parity 都写出来跑过，
cvxpy 与闭式解一致（max div 权重差 9.5e-6、`rho_LO`=0.405；risk parity 的
risk contribution 相对离散 9.9e-5）。`sigma`、`beta`、`omega2`、`sigmaM2` 四样齐全，
`rho_i = sigma_M * beta_i / sigma_i` 一行可得。**无需改动。**

**但 HRP 不行**：single-factor 下 correlation 的非对角部分精确是 rank-1，
clustering 没有 block 结构可发现 —— `rho_ij = rho_i * rho_j`，correlation distance
的排序完全由 `rho_i` 这一个标量决定，dendrogram 退化成一条按 `beta_i/sigma_i` 排的线。
要做 HRP 必须另给非 factor 的 `V`（CdST shrinkage 或 sample covariance）。

> 这一条是单独写脚本量的，不在 `min_risk_port.ipynb` 里：
> `max |C_ij - rho_i*rho_j| = 1.67e-16`（机器精度），非对角结构第二特征值 6.98e-15。
> max div / risk parity 的接口验证（权重差 9.5e-6 等）同样来自那次单独测试。

---

## 3. 关键决策

| 项 | 决定 | 理由 |
|---|---|---|
| Covariance | Single-factor，`V = sigmaM2 * beta beta' + Diag(omega2)` | project.pdf 对 topic 3 明文建议；HW4 的 closed form 可用 |
| 回归形式 | market model，**保留截距**，两边都是 excess return | 不是 CAPM；关心 covariance 结构，不强制 alpha = 0 |
| `omega2` 自由度 | `T - 2` | 估了 alpha 和 beta |
| Shrinkage | **只做 beta，Blume `2/3*beta_hat + 1/3`** | 对齐 HW3 Q2(c)；只保留一种 shrinkage |
| `sigmaM2` | 同一 T 窗口，`ddof=1` | 与 `se(beta_hat)` 用同一 `rM`，内部一致 |
| Max div 的 `sigma` | **必须取 `sqrt(diag(V))`（模型值）** | 否则 `sigma` 与 `V` 不一致，式 (4) 失效 |
| 生产求解器 | **闭式解**；cvxpy 留作交叉验证和带 cap 的情形 | 实测 3000 倍差距 |
| Solver 回退 | 不用 GUROBI；Clarabel → SCS → ECOS | peer reviewer 可能没有 license |
| 目标函数写法 | factor 形式 `sigmaM2*(beta'x)^2 + sum omega2_i x_i^2` | 更快，天然 PSD |
| Rebalance 频率 | **月度** | project.pdf (c) 允许降频但理由限定为 *"time constraints"*；闭式解让该前提不成立 |
| 缺失值 | `estimate_single_factor` 收到 NaN 直接报错 | universe 选择是上游职责，不做静默填补 |
| Delisting | `t+1` 无 return 的权重按 `rf` 计 | 按存活者归一化 = 月初就知道谁会消失（look-ahead） |
| Turnover | 对 **drift 后**的权重算，首期记 `NaN` | 否则把被动漂移算成主动交易 |

---

## 4. 实现中踩到并已解决的坑

### 4.1 cvxpy 的 interior-point solver 不产生精确的零

被排除的股票会拿到 dust，`n_held` 严重虚报（synthetic 上 40 → 68）。
`min_risk` 增加 `weight_tol=1e-5`，低于阈值截零后重新归一化。

**真实数据上这件事更微妙**：闭式解 42 只、cvxpy 45 只。
差的 3 只 beta 是 0.6458–0.6504，**刚好高于** `beta_LO = 0.6448` —— 本就该排除。
闭式解的目标值**严格更低**（cvxpy 差 1.7e-6 相对）。

**所以验证要比目标值，不是逐元素比权重。** notebook 的断言已改成：
目标值相对差 < 1e-4，且闭式解永不被 cvxpy 超越。

### 4.2 `mthcaldt` 是交易日月末，不是日历月末

1994-12-31 是周六 → 记录是 `1994-12-30`。所以 index 和 join **一律用 `yyyymm` 整数**。
MktRf.csv 的 index 正好也是 `yyyymm`，对得上。

### 4.3 `MktRf.csv` 是百分点，必须 ÷100

验证：60 月窗口的 `sigma_M = 0.0439`，落在 0.04–0.05 内。notebook 里有断言。

### 4.4 这份 extract 没有 share type 列

只有 `primaryexch`，做不了标准 CIZ 筛选。现在是 `primaryexch ∈ {N, A, Q}`，
丢掉 1.80% 的行。**这是个已知限制，slides 上要注明。**
（`mthret` 是小数；`mcap = mthcap / 1000`，VW 用哪个都一样，scale-invariant。）

---

## 5. 主结果（min risk，1995-01 .. 2025-12，T=60，n=500，372 个 OOS 月）

```
annualised return     8.20%
annualised vol       12.72%
Sharpe ratio          0.499
max drawdown        -37.55%
holds 45 of 500 stocks (effective N = 25), average beta 0.43
annualised turnover 116.62%
```

闭式解说的 **low-beta bet，实证完全对上**：平均只持 45/500 只，
`beta_LO` 均值 0.578（范围 0.437–0.707），ex-ante beta 均值 0.427。

### 子区间

| | ann_ret | ann_vol | Sharpe | avg_n_held | avg_beta |
|---|---|---|---|---|---|
| 1995-1999 | 0.0848 | 0.1308 | 0.307 | 51 | 0.489 |
| 2000-2009 | 0.0702 | 0.1340 | 0.373 | 57 | 0.336 |
| 2010-2019 | 0.1275 | 0.1052 | **1.149** | 35 | 0.456 |
| 2020-2025 | 0.0267 | 0.1448 | 0.069 | 36 | 0.480 |

Episode 累计收益：dot-com bust 2000-02 **+42.1%**、GFC 2007-09 -37.1%、
COVID 2020 -15.6%、2022 **+4.3%**。

> Episode 用**累计**不用年化：COVID 只有 2 个月，年化会得到 -80%、Sharpe -14.5
> 这种看着吓人但没信息量的数字。

**子区间必须把 2000-02 单独拿出来。** 收窄范围前测过 VW 的对照：
同期 VW 是 **-34.8%**（min risk +42.1%，差 **76.9pp**），而 1995-1999 期间
min risk 年化 +8.5%、VW +28.9%，min risk 大输。一个 1995-2007 的桶会把两者抵消掉。

---

## 6. 最强的发现：模型低估 min risk 的风险，且原因不是集中度

`predicted / realized vol` 对 min risk 只有 **0.574**。

这个数**单看没有意义** —— 完全可能是 scaling 或 annualization 错了。
所以用**随机 long-only 组合**做对照：同一个 universe、同一个 covariance model 定价。

| | mean ratio | range |
|---|---|---|
| **Minimum risk** | **0.5741** | — |
| Random, diffuse（500 只随机权重） | 0.9688 | 0.9650–0.9732 |
| **Random, concentrated（随机 45 只，组内 `1/omega2` 加权）** | **0.9740** | 0.9489–0.9932 |

**第二行对照是关键**：随机抽 45 只、**和 min risk 一样集中**，模型定价准到 97%。
所以问题不在"持股少"，而在**优化器挑的是哪些股票** —— 它系统性地选中 residual
正相关的那批低 beta 防御股，而 single-factor 假设 residual 不相关。

这是"要么换更丰富的 covariance，要么加个股上限"的直接论据，论证链干净，可直接进 slides。

---

## 7. 从 project.pdf 读出来的范围判断

逐句量过 topic 3 之后：

- **强制交付面很小**：五个组合 + 一张 Exhibit 1 风格的表 + 一张 Exhibit 4 风格的图
- **turnover 和个股上限都是 topic 1 的句子**，topic 3 从头到尾没提 —— 自选项，不是要求
- **covariance zoo 是 topic 1 的得分面**。topic 1 的 Goal 句就是
  *"via various approaches to covariance estimation"*；topic 3 的 Goal 句是
  diversification 比较，对 covariance 只说 *"feel free to use other"* —— 许可，不是奖励
- **PDF 点名的唯一 robustness 轴是 test periods**
  （*"different test periods that span different kinds of economic conditions"*），
  不是 covariance estimator，也不是 n
- **HW4 式 (3)(4)(5) 就是 CdST 2013 的解析解**（notation `sigma2_LMV` / `beta_LO` /
  `rho_LO` / `sigma_LA` / `gamma` 一致）。instructor 指定那篇当表格范本，
  用意应是把解析结构和 out-of-sample 结果连起来：
  min risk 只持 `beta_i < beta_LO` 是 low-beta bet；
  max div 只持 `rho_i < rho_LO` 是 low-correlation bet；risk parity 全持有
- 教材 `OIFbook2026.pdf` §7.5 原文给了统一视角：max div 的 diversification ratio
  *"is proportional to the Sharpe ratio if mu is proportional to sigma"*，
  即各组合都是某套 implicit return assumption 下的 max-Sharpe

---

## 8. 范围收窄：EW / VW 已移除

按分工，EW / VW 不归本部分。`portfolios.py` 现在只 export
`min_risk` 和 `min_risk_closed_form`；`backtest.py` 的
`CONSTRUCTORS = {"min_risk": min_risk_closed_form}`。
（`rolling_backtest` 仍把 `caps=` 透传给 constructor，因为 `select_universe`
要用 caps 选 universe，也保留扩展性。）

> 两个被删的函数各只有 3 行，在提交 `956e4a1` 里，需要时随时能捞回来。

收窄带来两个损失，都已补上：

1. **EW 的 sanity check 没了**（原来"EW return 等于截面简单平均"是对整条 pipeline
   的独立验证）。替代：从存下的权重和原始 return panel **独立重算**三个月份的组合收益
   （首期 / 中间 / 末期），与 backtest 自记的 `ret` 对齐。强度等价。
2. **predicted/realized 的对照组没了**。替代见第 6 节的随机组合对照 ——
   **比原来的 EW/VW 对照更强**，因为它能区分"集中度"和"选股"两个假设。

---

## 9. 待定 / 下一步

- [ ] 是否加 residual volatility 的 log-vol shrinkage（目前未做）
- [ ] 是否加 Vasicek beta shrinkage 作为对比
- [ ] 是否加 cross-sectional winsorize。**注意**：截断原始 return 会在市场大幅波动的
      月份系统性削掉 high beta 股票（截面离散度随 |market move| 放大，高 beta 落在尾部），
      与 Blume shrinkage 叠加成双重收缩。若要做，建议截断 **residual** 而非原始 return
- [ ] 是否加个股上限 3%/5%（会破坏 closed form，只能走 cvxpy）。
      第 6 节的发现是支持加上限的直接论据
- [ ] shrinkage 的 ablation（不 shrink → 只 shrink beta → beta + omega）

### 和队友对接时必须说的

**跨组合的比较（min risk vs EW vs VW）现在没有人在同一套 universe 和 covariance 下做。**
谁来做都行，但必须保证三点口径一致，否则表没法并排放：

1. 同一个 universe 选择规则（top-500 by cap + complete 60m history + `primaryexch ∈ {N,A,Q}`）
2. 同一个 `rf`（MktRf.csv ÷100）
3. delisting 口径一致（`t+1` 无 return 的权重按 `rf` 计）

### 打包约束

project.pdf 只允许「1 个 notebook + 最多 1 个 `.py` + 1 个 `.csv`」，
现在是 3 个 `.py`，**最终交付前要合并**。
