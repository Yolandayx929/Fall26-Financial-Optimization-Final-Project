# Data + Benchmarks + Final Comparison —— 进度与决策记录

> 本人负责 topic 3 的 **数据**（下载、清洗、data-quality check、universe summary）、
> 两个 benchmark（**EW / VW**），以及全组的 **最终对比**（汇总表 + 累积收益图）。
> 不改动队友的 `estimation.py` / `min_risk.py` / `risk_parity.py` / `backtest.py` / `backtesting_analysis.py`，
> 只新增文件。统计、金融术语用 English，其余用中文。

---

## 1. 新增文件

| 文件 | 内容 |
|---|---|
| `project_data.py` | `download_crsp`（从 WRDS 下载并生成 `data/crsp_msf_v2.parquet`）、`data_quality_report`、`universe_summary` |
| `benchmarks.py` | `equal_weight`、`value_weight`，两个 backtest constructor |
| `analysis/comparison.ipynb` | 全组最终对比：所有策略一起跑 backtest，出汇总表和累积收益图 |

---

## 2. 数据是怎么来的

**一句话：** 从 WRDS 拉 CRSP 月度数据，只留美国普通股，存成 `data/crsp_msf_v2.parquet`，大家都读这一个文件。

- **来源：** `crsp.msf_v2`（CRSP 新版 CIZ 格式，年度更新），1990-01 到 2025-12。
- **只要美国普通股：** 在下载的 SQL 里就筛好了（`sharetype='NS'`、`securitytype='EQTY'`、`securitysubtype='COM'`、`usincflg='Y'`、`issuertype` 为 `ACOR`/`CORP`）。
  ETF、基金、ADR（外国公司在美上市）都不在文件里。
  *所以 notebook 里不需要再做 common-share filter，文件已经是普通股了。*
- **交易所：** 只留 NYSE / NYSE American / Nasdaq（`primaryexch` 为 N、A、Q），这一步在 notebook 里加载后做。
- **退市收益：** CIZ 格式的 `mthret` 已经把退市当月的收益算进去了，不用另外处理。
- **市值：** `mthcap`，单位是千美元。
- **Mkt-RF 和 RF：** 来自 `data/MktRf.csv`（Kenneth French），原始单位是百分点，读进来要除以 100。

**peer reviewer 怎么复现：** 如果 `data/crsp_msf_v2.parquet` 不存在，`comparison.ipynb` 会用他们自己的 WRDS 账号重新下载，得到完全一样的文件。

---

## 3. Data-quality check 的结果

| 检查项 | 结果 | 说明 |
|---|---|---|
| 总行数 / 股票数 | 2,179,752 行 / 18,905 只 | 1990-01 到 2025-12 |
| 同一股票同一月重复 | 0 | 干净 |
| `mthret` 缺失 | 1.83% | 不影响结果：进投资池要求过去 60 个月收益都不缺 |
| 收益 < -100% | 0 | 不可能的值没有出现 |
| 收益 > +300% | 526 条 | 抽查过最大的几条（如 GME 2021-01 +1625%），都是真实事件，不是数据错误，所以**不做 winsorize** |
| 不在 N/A/Q 交易所 | 1.80% | notebook 里删掉 |

---

## 4. 投资池（universe）

**按全组统一的规则：** 用 `backtest.select_universe`。每个月 t 月底，在"过去 60 个月收益都不缺、t 月市值 > 0"的股票里，取市值最大的 500 只。

`universe_summary` 每个月统计一次（1995–2025，372 个月的平均）：

| 指标 | 平均值 | 大白话 |
|---|---|---|
| 合格股票数 | 约 3,250 只 | 能进候选池的股票 |
| 第 500 名的市值 | 约 51 亿美元 | 进前 500 的门槛 |
| 占全市场总市值 | 约 80% | 500 只股票覆盖了美股八成市值 |
| 前 10 大在 VW 里的权重 | 约 23% | 市值加权有多集中 |
| 每月新进股票 | 约 12 只 | 投资池每月换掉一点点 |
| 下个月没有收益的股票 | 约 1.5 只 | 按全组约定，这部分权重**按 RF 计收益**（当作持有现金） |

**已知限制（讨论后决定不处理）：** 下个月没收益的那约 1.5 只，大多是 t 月当月就因并购退市的股票，月底其实已经买不到了。
全组决定不单独剔除它们，对结果影响很小。

---

## 5. 两个 benchmark

| 名字 | 怎么定权重 | 代码 |
|---|---|---|
| EW（equally weighted） | 500 只每只 1/500 | `equal_weight(cov)` |
| VW（value weighted） | 按 t 月底市值占比 | `value_weight(cov, caps)` |

- 两个都写成和 `risk_parity` 一样的 **constructor**：输入 `cov`，返回 `dict(x, sigma2, n_held)`，直接放进 `rolling_backtest(constructors=...)`。
- 所以 EW / VW 和优化组合用的是**同一个 universe、同一套 turnover 和缺失收益口径**，比较才公平。
- `cov` 不参与定权重，只用来报告 predicted variance，让 benchmark 也能出现在同一张表里。

---

## 6. 最终对比（`analysis/comparison.ipynb`）

**流程：** 读数据 → data-quality check → universe summary → 所有策略一起跑一次 `rolling_backtest` → `check_alignment`（确认月份、universe、权重都对齐）→ `performance_table` 出汇总表（并和 `backtest.summarize` 交叉核对）→ 累积收益图。

**参数：** 1995-01 到 2025-12 out-of-sample（372 个月），T = 60，n = 500，每月 rebalance。

**指标口径**（沿用 `backtesting_analysis.py`）：年化收益用几何平均；vol 和 Sharpe 用 excess return；max drawdown 用复利路径；turnover 是单边，对比上月 drift 之后的权重；beta 对 Mkt-RF。

### 结果

| | ann_return | ann_vol | Sharpe | max_drawdown | ann_turnover | beta | 1 美元变成 |
|---|---|---|---|---|---|---|---|
| EW | 11.63% | 15.7% | 0.63 | -51.7% | 62% | 0.98 | $30.3 |
| VW | 11.55% | 14.8% | 0.66 | -49.1% | 10% | 0.94 | $29.6 |
| Minimum risk | 8.20% | 12.7% | 0.50 | -37.6% | 117% | 0.34 | $11.5 |
| Risk parity | 11.62% | 14.0% | 0.69 | -48.9% | 62% | 0.85 | $30.2 |

**大白话解读：**
- **Minimum risk 真的最稳：** vol 最低、回撤最小、beta 只有 0.34。但这 30 年里收益也最低，Sharpe 反而最差，而且换手最高。
- **Risk parity 的 Sharpe 最高：** 收益和 EW 差不多，但波动更小。
- **EW 和 VW 很接近：** VW 换手只有 10%，几乎不用交易，是最便宜的基准。

累积收益图是对数坐标，每条线代表 1 美元的增长，最终金额写在图例里。

---

## 7. 加新策略

在 `comparison.ipynb` 第 4 节的 `STRATEGIES` 字典里加一行 `"名字": constructor`，然后从头重新运行。汇总表和累积收益图会自动多出这个策略。

constructor 的格式和 `risk_parity` 一样：输入 `cov`（`FactorCov`），可选 `caps`，返回 `dict(x=权重, sigma2=预测方差, n_held=持有数)`。
