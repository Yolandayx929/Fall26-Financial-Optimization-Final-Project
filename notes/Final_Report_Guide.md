# Final report / Google Slides guide

核心研究和整合 demo 已完成；项目还需完成匿名 slides PDF、按课程入口提交与个人 peer review。

## project.pdf 的硬性要求

- Topic 3：fully invested、long only 的 minimum risk / maximum diversification / risk parity，与 EW 和 VW 共同做 out-of-sample 比较。
- 摘要统计表：参照 Clarke / de Silva / Thorley (2013) Exhibit 1；累计收益图：参照同作者 (2006) Exhibit 4。
- Highlights：匿名、最多 10 页、适合 10 分钟演讲；交付 PDF，而不只是 Google Slides 链接。
- Demo：理想为一个 notebook；最多可再带一个 `.py` 与一个 CSV；多文件压成一个 ZIP。
- 初稿：2026-10-02 午夜前；终稿：2026-10-05 中午前；个人 peer review / 投票：2026-10-06 08:00 前。PDF 没明确时区，以 Canvas 显示为准。

## 已准备好的交付物

`final/project_demo.ipynb`、`final/project.py`、`final/MktRf.csv`。提交时只将这三个文件压缩为 `final_demo.zip`。
ZIP 不包含 CRSP 原始数据、账号、个人姓名或仓库链接。解压后 notebook 会读取本地 cache，
没有 cache 时引导用户以自己的 WRDS 账号下载。安装说明在 notebook 第一部分。
`final/report_results/` 是本次运行导出的 slides 素材，不需要放进 demo ZIP。

## 建议用 10 页，英文标题可直接贴进 Google Slides

| 页 | 英文标题 | 放什么 / 要讲什么 | 时间 |
|---|---|---|---|
| 1 | Diversification Is More Than Holding More Stocks | Topic 3、研究问题；一句结论：RP 在本样本 Sharpe 最高，min-risk 最防御，MaxDiv 的高估计 DR 没转换成最高实际 Sharpe。不放姓名、学号、团队链接。 | 0:35 |
| 2 | One Universe, One Backtest, Five Portfolios | CRSP 1990–2025；测试 1995–2025；500 最大合资格股票；60 月完整历史；每月再平衡；French 数据 ÷100。画一条 t 估计 / t+1 计收益的时间轴。 | 0:55 |
| 3 | Five Rules for Allocating the Same Capital | EW、VW、min x'Vx、max DR、equal total RC 的五行表；共同约束 sum(x)=1、x≥0；单因子 + Blume beta shrinkage；无需展开代码或长推导。 | 1:05 |
| 4 | Risk Parity Has the Highest Full-Sample Sharpe | `performance_table.png`；强调 RP Sharpe 0.689，min-risk vol 12.72%，MaxDiv CAGR 10.35%。说明 geometric total CAGR 和 arithmetic excess return 是不同的指标。 | 1:15 |
| 5 | Lower Risk Does Not Guarantee Higher Growth | `cumulative_returns.png`；五条线、相同基准、1995–2025；明确是 compounded total growth of $1、log scale。EW/RP/VW 的末期值接近，不说谁稳定跑赢。 | 1:00 |
| 6 | Minimum Risk Offers the Strongest Drawdown Protection | `drawdowns.png`；min-risk 最大回撤 −37.55%，MaxDiv −53.62%；与同一期间 EW/VW/RP 比较，区分 volatility 和 drawdown。 | 0:55 |
| 7 | The Preferred Strategy Changes Across Regimes | `subperiod_sharpe.png`；单列 2000–2002，避免十年桶掩盖 dot-com；2007–2009 单列；挑 2–3 个时期讲，不逐格念。 | 1:00 |
| 8 | Maximum DR and Equal Risk Contributions Mean Different Things | `concentration.png` + `concentration.csv` 中的 DR 小表；MaxDiv DR 3.661、平均 63 股；RP 500 股且有效股票风险数 500，但单因子风险占比约 98.93%，不是 500 个独立风险来源。 | 1:10 |
| 9 | Turnover and Model Risk Limit the Gross Results | `turnover.png`；VW 10.01%、MaxDiv 160.36% 年度 one-way turnover；未扣交易成本，不推断净收益排名。说明完整历史偏向老股票、missing next-month return 用 RF、单因子忽略 residual cross-correlation。 | 1:00 |
| 10 | Match the Diversification Rule to the Investment Objective | 结论三点：本样本 RP 风险收益折衷最好；min-risk 更防御但回报较低；MaxDiv 只优化估计 DR。小字文献、demo 可复现方法和未来改进，不放个人代码仓库链接。 | 1:05 |

总计约 10 分钟。每页一条 takeaway；图表轴与数值保留清晰标签；不用贴 notebook 截图。

## 可以直接使用的真实结果

| Strategy | CAGR % | Vol % | Sharpe | Max DD % | Turnover % | Beta |
|---|---:|---:|---:|---:|---:|---:|
| Value weighted | 11.55 | 14.82 | 0.656 | -49.13 | 10.01 | 0.944 |
| Equally weighted | 11.63 | 15.74 | 0.631 | -51.70 | 61.66 | 0.977 |
| Minimum risk | 8.20 | 12.72 | 0.499 | -37.55 | 116.62 | 0.339 |
| Maximum diversification | 10.35 | 14.11 | 0.604 | -53.62 | 160.36 | 0.673 |
| Risk parity | 11.62 | 14.04 | 0.689 | -48.90 | 62.41 | 0.846 |

注：CAGR 为 geometric annual total return；Vol 与 Sharpe 使用 excess returns；Beta 是对 Mkt-RF 的 realized 回归 beta。
Max DD 从含初始本金 1 的累计财富路径计算；Turnover 为 mean monthly one-way drift-adjusted turnover ×12，
不是双边成交额、交易成本或首次建仓成本。

## 解读边界

- RP 的均衡是股票的 total risk contribution；同一股票市场里的共同市场风险仍可高度集中。
- MaxDiv 最大化 model-implied DR，没有预测 expected returns；高 DR 不保证最高 realized Sharpe/return。
- Min-risk 的低 realized beta 不等于其每月 ex-ante beta；不要把两者混用。
- 各策略的实际排名是本样本描述，不是“统计显著跑赢”或未来收益保证；本项目没有做显著性检验。
- “优化器系统性选中某些行业 / residual 正相关导致低估风险”需要独立证据；本 deck 不把未验证归因写成定论。
- CRSP 缓存的分类字段未保留；SQL 包含普通股筛选，但不能仅凭缓存独立证明其来源。
- 无需补 HRP、更多 covariance 或个股上限来满足 Topic 3。匿名 slides 和交付是剩下的必要步骤。

## 复核与复现记录

整合包在仓库外的独立临时目录运行，没有导入原仓库模块；使用既有 CRSP cache。
372 月 × 5 策略运行成功；首期 min-risk / MaxDiv QP 与 RP convex 对照通过；
共享绩效交叉检查、所有日期 universe/weights 对齐、MaxDiv DR 优势和 KKT 检查通过；
全样本 RP contribution dispersion 检查通过。原模块与整合模块首期三种优化策略权重一致。
独立目录完整运行约 22 秒（不含网络下载；机器与 solver 不同会改变耗时）。
本环境使用 Python 3.10.9、numpy 1.23.5、pandas 2.2.3、pyarrow 25.0.1。
没有使用真实 WRDS 凭据重下载；fresh-download 分支通过 fake-DB 测试，不代表已验证真实网络授权。

## 提交前最后一步

1. 按上面的提纲完成 Google Slides；确认匿名、≤10 页，导出 PDF 并检查字体/图例。
2. 将 highlights PDF 和 `final_demo.zip` 按 Canvas 的作业入口提交。
3. 用有 CRSP 权限的同学账号做一次 fresh WRDS download（可选本地清空缓存路径），确认真实登录与权限。
4. 每个人完成 10 月 6 日早上截止的 peer review 与投票。

## References for slide footnotes

- Clarke, de Silva and Thorley (2013), Risk Parity, Maximum Diversification, and Minimum Variance: An Analytic Perspective. JPM 39(3), 39–53. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1977577
- Clarke, de Silva and Thorley (2006), Minimum-Variance Portfolios in the U.S. Equity Market. JPM 33(1), 10–24. DOI: 10.3905/jpm.2006.661366.
- `project.pdf`, pp. 1–2 and 4；CRSP msf_v2 via WRDS；Kenneth French market/risk-free data (course MktRf.csv).

表格按 2013 Exhibit 1 的跨策略方式呈现，增加清楚标注的 CAGR / drawdown / turnover；
图采用 2006 Exhibit 4 要求的跨组合累计收益比较思路，不宣称复刻原论文的数据、sample period 或所有估计方法。
