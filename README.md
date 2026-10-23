# US ETF Predictor MVP

面向个人研究的美股 ETF 概率预测项目。它输出未来 5 个交易日上涨概率和样本外评估，**不连接券商、不生成订单、不执行交易**。

## 第一版范围

- 标的池：`SPY`、`QQQ`、`IWM`、`XLK`、`XLF`、`XLE`、`TLT`、`GLD`、`SOXX`
- 预测目标：默认预测 `SPY`、`QQQ`、`IWM` 未来 5 个交易日方向
- 特征：价量因子（收益率、波动率、均线偏离、RSI、成交量异常、跨 ETF 联动），以及宏观、美联储议息、基本面、新闻情绪因子
- 模型：优先使用 LightGBM，运行库不可用时回退 Extra Trees；按时间切分并隔离预测周期
- 数据：按种类划分的独立数据层，默认全部离线合成数据；可逐类切换真实数据源（见下文）
- 存储：Parquet + DuckDB
- 展示：Streamlit 多页面应用，包含交互预测看板与项目技术说明页
- 分钟模型：仅为 `QQQ`、`SOXX` 拉取多年 5 分钟行情，各自训练独立的序列特征模型并单独走步回测

## 快速开始

要求 Python 3.11 或 3.12，推荐使用 `uv`：

```bash
uv sync --extra dev
uv run etf-mvp run --provider synthetic
uv run streamlit run app.py
```

浏览器打开 Streamlit 输出的本地地址即可查看预测、样本外指标和特征重要性。

看板左侧可以选择预测标的、预测周期、判断阈值和数据模式；点击“运行研究模型”即可刷新结果。行情页支持 K 线和多 ETF 归一化走势对比。

侧栏的“项目说明”页面展示当前系统的数据、特征、模型、评估、产物和工程边界，便于快速理解架构与后续扩展方向。

## 数据层

数据处理已抽成独立的 `etf_mvp.data` 包，按**数据种类**组织，每类数据有统一的结构和独立的数据源：

| 种类 (`kind`) | 内容 | 真实数据源 | 所需环境变量 |
|---|---|---|---|
| `market` 行情 | 日线复权 OHLCV | Tiingo | `TIINGO_API_KEY` |
| `macro` 宏观 | 利率、利差、VIX、信用利差、CPI、失业率、初请 | FRED / ALFRED | `FRED_API_KEY`（免费） |
| `fundamental` 基本面 | ETF 主要成分股的营收、净利润、EPS、权益 | SEC EDGAR XBRL | `SEC_USER_AGENT`（姓名 + 邮箱） |
| `news` 新闻 | 新闻标题、主题、情绪分 | Alpha Vantage NEWS_SENTIMENT | `ALPHAVANTAGE_API_KEY`（免费） |
| `central_bank` 美联储议息 | FOMC 日程、利率决议、鹰鸽分（决议日 2 年期美债收益率变动） | 内置会议日历 + FRED | `FRED_API_KEY`（免费） |

分钟级数据不进入上述共享日线数据层。`QQQ` 与 `SOXX` 使用 Alpaca Historical Bars API 分别存储和训练，避免把两个标的混成同一个模型。Basic 账户使用免费的 `iex` 历史数据，并避开最近 20 分钟；订阅相应套餐后可显式改用 `--feed sip`。

每一类都另有 `synthetic` 离线数据源，供开发和测试使用。

**避免前视偏差**：除行情外，每条记录都带 `available_date`（首次公开、可在当日收盘使用的日期），因子只按 `available_date <= 交易日` 对齐：

- 宏观：会修订的序列（CPI、失业率、初请）用 ALFRED 的首次发布值和真实发布日。
- 基本面：取每个报告期首次申报的数值，申报日次一交易日起可用。
- 新闻：美东 16:00 后发布的新闻计入下一交易日。
- 议息：会议日程提前公布，可直接使用；利率决议取 FRED 目标区间（DFEDTARU/DFEDTARL），鹰鸽分为决议当天 2 年期美债收益率（DGS2）的变动（基点，正为偏鹰）。DGS2 次一交易日才在 FRED 发布，所以决议记录从反应日的下一交易日起可用。

**已知局限**：基本面用的是配置里的成分股快照，回溯历史会带幸存者偏差；Alpha Vantage 免费层每天约 25 次请求、新闻约从 2022 年开始，新闻数据按增量方式分多次补齐；美联储会议日历内置在 `src/etf_mvp/data/reference/fomc_meetings.csv`（2014–2027），每年美联储公布新日程后需手动追加，也可在 `data/reference/fomc_meetings.csv`（列 `event_date,is_scheduled`）补充或覆盖。

为防止把合成数据和真实行情混在一起训练，行情使用真实数据源时，仍配置为 `synthetic` 的其他种类会被自动跳过。

### 使用真实数据源

```bash
cp .env.example .env   # 填好后 CLI 与 Streamlit 会自动读取

uv run etf-mvp sources                                   # 查看各类数据源配置
uv run etf-mvp --config configs/live.toml run            # 全部真实数据
uv run etf-mvp --config configs/live.toml fetch --kind macro --kind central_bank
uv run etf-mvp run --provider tiingo --source macro=fred # 在默认配置上逐类切换

# 仅拉取 QQQ、SOXX 多年 5 分钟数据
uv run etf-mvp --config configs/free.toml intraday-fetch --start 2018-01-01 --feed iex

# 使用已有分钟数据分别训练并回测 QQQ、SOXX
uv run etf-mvp --config configs/free.toml intraday-train

# 拉取、独立训练、独立回测一次完成
uv run etf-mvp --config configs/free.toml intraday-run --start 2018-01-01 --feed iex
```

不要提交真实 API Key。

## 常用命令

```bash
# 只拉取并持久化数据（全部种类，或用 --kind 指定）
uv run etf-mvp fetch
uv run etf-mvp fetch --kind news

# 使用已有数据重新训练
uv run etf-mvp train

# 运行测试与静态检查
uv run pytest
uv run ruff check .
```

## 产物

- `data/market/prices.parquet`：标准化 OHLCV
- `data/raw/<kind>/<provider>.parquet`：其他各类原始数据（带 `available_date`）
- `data/processed/features.parquet`：模型特征
- `data/warehouse.duckdb`：数据仓库，每类数据一张表，外加 `model_features`
- `artifacts/predictions.csv`：最新概率预测
- `artifacts/metrics.json`：时间切分后的样本外指标
- `artifacts/models/*.joblib`：每个目标 ETF 的模型
- `artifacts/importance_by_kind.csv`：各类数据对模型的贡献
- `data/intraday/QQQ_5min.parquet`、`SOXX_5min.parquet`：隔离存储的真实 5 分钟行情
- `artifacts/intraday/<symbol>/forecast.csv`：QQQ/SOXX 各自的下一交易日 78 时点预测
- `artifacts/intraday/<symbol>/backtest_sessions.csv`：分钟模型独立扩展窗口回测
- `artifacts/intraday/<symbol>/model.joblib`：每个标的独立训练的分钟模型

## 成本与边界

本地运行和合成数据成本为 0；Tiingo 免费层是否满足个人用途以其最新条款为准。不需要 GPU、LLM 或实时行情。模型结果仅用于研究，不构成投资建议。
