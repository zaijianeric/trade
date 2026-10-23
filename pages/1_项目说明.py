from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="项目说明 · ETF Research Lab", page_icon="📘", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #08121f; }
    [data-testid="stHeader"] { background: rgba(8, 18, 31, 0.92); }
    [data-testid="stSidebar"] { background: #101827; }
    [data-testid="stMainBlockContainer"] { max-width: 1500px; padding-top: 2.3rem; }
    h1, h2, h3, h4 { color: #e7eef8 !important; }
    p, li { color: #a9b8ca; }
    code { color: #7dd3fc !important; }

    .doc-hero {
        display: flex;
        justify-content: space-between;
        gap: 28px;
        align-items: flex-end;
        margin-bottom: 28px;
    }
    .doc-title {
        margin: 0;
        color: #edf4ff;
        font-size: clamp(2rem, 4vw, 3.7rem);
        line-height: 1.08;
        font-weight: 800;
        letter-spacing: -.04em;
    }
    .doc-subtitle { margin: 12px 0 0 0; font-size: 1rem; color: #8fa4ba; }
    .flow-label {
        color: #58c6d3;
        font-size: 1rem;
        letter-spacing: .03em;
        white-space: nowrap;
    }

    .architecture-shell {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 250px;
        gap: 18px;
        align-items: stretch;
    }
    .architecture-rows { display: grid; gap: 12px; }
    .architecture-row {
        display: grid;
        grid-template-columns: 150px repeat(3, minmax(0, 1fr));
        gap: 12px;
    }
    .layer-label,
    .architecture-card,
    .governance-panel {
        border: 1px solid #293b4e;
        border-radius: 13px;
        background: #111d29;
    }
    .layer-label {
        padding: 17px 18px;
        background: #162534;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .layer-number { color: #61c4d1; font-size: 1rem; font-weight: 800; }
    .layer-name { color: #e7eef8; font-size: 1rem; font-weight: 750; margin-top: 4px; }
    .architecture-card { padding: 17px 19px; min-height: 84px; }
    .card-title { color: #e5edf7; font-size: 1rem; font-weight: 750; }
    .card-detail { color: #95a8bb; font-size: .92rem; margin-top: 4px; line-height: 1.4; }
    .architecture-row.model .layer-label,
    .architecture-row.output .layer-label { background: #103943; border-color: #235865; }
    .architecture-row.model .architecture-card,
    .architecture-row.output .architecture-card { border-color: #41a9b7; background: #102a31; }
    .architecture-row.boundary .layer-label { background: #432d1a; border-color: #654422; }
    .architecture-row.boundary .architecture-card { border-color: #d88b31; background: #281d14; }
    .architecture-row.boundary .layer-number { color: #f3a64e; }

    .governance-panel { padding: 24px 20px; background: #14212e; }
    .governance-title { color: #5cc5d2; font-size: 1.3rem; font-weight: 800; margin-bottom: 16px; }
    .governance-item {
        background: #0d1722;
        border-radius: 10px;
        padding: 13px 14px;
        margin-bottom: 11px;
    }
    .governance-item strong { display: block; color: #edf4ff; font-size: .94rem; }
    .governance-item span { color: #91a4b7; font-size: .84rem; }
    .feedback-loop { color: #f2a34b; font-weight: 750; margin-top: 22px; line-height: 1.45; }

    .section-label {
        color: #5cc5d2;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin: 38px 0 8px;
    }
    .doc-panel {
        background: #111d29;
        border: 1px solid #293b4e;
        border-radius: 14px;
        padding: 20px 22px;
        height: 100%;
    }
    .doc-panel h3 { margin-top: 0; font-size: 1.1rem; }
    .doc-panel p { margin-bottom: 0; line-height: 1.65; }
    .stack-chip {
        display: inline-block;
        padding: 5px 10px;
        margin: 3px 4px 3px 0;
        border-radius: 999px;
        color: #b9e9ef;
        background: #12323a;
        border: 1px solid #275e68;
        font-size: .82rem;
    }
    .boundary-note {
        border-left: 4px solid #f0a044;
        background: #261d15;
        color: #f5cf9e;
        padding: 14px 16px;
        border-radius: 8px;
        margin-top: 18px;
    }
    @media (max-width: 1050px) {
        .architecture-shell { grid-template-columns: 1fr; }
        .architecture-row { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .layer-label {
            grid-column: 1 / -1;
            flex-direction: row;
            align-items: center;
            gap: 10px;
        }
        .architecture-card { min-height: auto; }
        .doc-hero { display: block; }
        .flow-label { margin-top: 18px; white-space: normal; }
    }
    @media (max-width: 680px) {
        .architecture-row { grid-template-columns: 1fr; }
        .layer-label { grid-column: auto; }
    }
    </style>

    <div class="doc-hero">
      <div>
        <h1 class="doc-title">ETF 量化预测 MVP · 技术架构</h1>
        <p class="doc-subtitle">面向个人研究的美股 ETF 日线方向概率系统</p>
      </div>
      <div class="flow-label">数据 → 特征 → 模型 → 概率 → 看板</div>
    </div>

    <div class="architecture-shell">
      <div class="architecture-rows">
        <div class="architecture-row">
          <div class="layer-label"><span class="layer-number">01</span><span class="layer-name">数据源</span></div>
          <div class="architecture-card"><div class="card-title">日线与分钟</div><div class="card-detail">Nasdaq 日线 · Alpaca QQQ/SOXX 5分钟</div></div>
          <div class="architecture-card"><div class="card-title">新闻情绪</div><div class="card-detail">SF Fed · Daily News Sentiment</div></div>
          <div class="architecture-card"><div class="card-title">地缘风险</div><div class="card-detail">GPR · Threats · Acts</div></div>
        </div>

        <div class="architecture-row">
          <div class="layer-label"><span class="layer-number">02</span><span class="layer-name">接入与存储</span></div>
          <div class="architecture-card"><div class="card-title">统一数据接口</div><div class="card-detail">Provider 抽象 · 标准化 date/symbol/OHLCV</div></div>
          <div class="architecture-card"><div class="card-title">列式文件</div><div class="card-detail">Parquet · 行情与特征持久化</div></div>
          <div class="architecture-card"><div class="card-title">分析仓库</div><div class="card-detail">DuckDB · market_prices / model_features</div></div>
        </div>

        <div class="architecture-row">
          <div class="layer-label"><span class="layer-number">03</span><span class="layer-name">特征工程</span></div>
          <div class="architecture-card"><div class="card-title">行情技术</div><div class="card-detail">收益 · 波动 · 均线 · RSI · 成交量</div></div>
          <div class="architecture-card"><div class="card-title">新闻与地缘</div><div class="card-detail">情绪水平/变化 · GPR/威胁/行动</div></div>
          <div class="architecture-card"><div class="card-title">宏观与央行</div><div class="card-detail">利率 · 通胀 · 就业 · FOMC</div></div>
        </div>

        <div class="architecture-row model">
          <div class="layer-label"><span class="layer-number">04</span><span class="layer-name">模型训练</span></div>
          <div class="architecture-card"><div class="card-title">树与序列模型</div><div class="card-detail">日线分类 · 分钟直接多时点回归</div></div>
          <div class="architecture-card"><div class="card-title">时间验证</div><div class="card-detail">扩展窗口 · 标的独立回测</div></div>
          <div class="architecture-card"><div class="card-title">概率评估</div><div class="card-detail">ROC AUC · Brier · Log Loss · Accuracy</div></div>
        </div>

        <div class="architecture-row output">
          <div class="layer-label"><span class="layer-number">05</span><span class="layer-name">预测服务</span></div>
          <div class="architecture-card"><div class="card-title">研究输出</div><div class="card-detail">上涨概率 · UP / DOWN · 重要特征</div></div>
          <div class="architecture-card"><div class="card-title">模型产物</div><div class="card-detail">CSV · JSON · Joblib</div></div>
          <div class="architecture-card"><div class="card-title">交互看板</div><div class="card-detail">Streamlit · Plotly · 参数化重训</div></div>
        </div>

        <div class="architecture-row boundary">
          <div class="layer-label"><span class="layer-number">06</span><span class="layer-name">项目边界</span></div>
          <div class="architecture-card"><div class="card-title">不执行交易</div><div class="card-detail">无券商连接 · 无账户权限 · 无订单接口</div></div>
          <div class="architecture-card"><div class="card-title">研究模型</div><div class="card-detail">分钟预测非实时报价 · 非高频系统</div></div>
          <div class="architecture-card"><div class="card-title">决策辅助</div><div class="card-detail">概率不是收益承诺 · 不构成投资建议</div></div>
        </div>
      </div>

      <aside class="governance-panel">
        <div class="governance-title">工程质量与治理</div>
        <div class="governance-item"><strong>环境可复现</strong><span>uv.lock · Python 3.12</span></div>
        <div class="governance-item"><strong>质量检查</strong><span>pytest · Ruff</span></div>
        <div class="governance-item"><strong>防止泄漏</strong><span>时间切分 · Horizon Gap</span></div>
        <div class="governance-item"><strong>模型追踪</strong><span>Metrics · Feature Importance</span></div>
        <div class="governance-item"><strong>数据安全</strong><span>API Key 仅来自环境变量</span></div>
        <div class="governance-item"><strong>经济成本</strong><span>本地 CPU · 免费层可启动</span></div>
        <div class="feedback-loop">↑ 样本外反馈 → 特征和模型迭代</div>
      </aside>
    </div>
    """.replace("\n\n", "\n"),
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Project Overview</div>', unsafe_allow_html=True)
overview_left, overview_middle, overview_right = st.columns(3)
with overview_left:
    st.markdown(
        """
        <div class="doc-panel">
          <h3>目标</h3>
          <p>预测指定美股 ETF 在未来 1–20 个交易日的上涨概率，验证特征与模型是否具有稳定的样本外解释力。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with overview_middle:
    st.markdown(
        """
        <div class="doc-panel">
          <h3>当前版本</h3>
          <p>第一版 MVP 聚焦日线数据、树模型和可视化验证，以低成本、快速迭代和可复现为优先级。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with overview_right:
    st.markdown(
        """
        <div class="doc-panel">
          <h3>适用场景</h3>
          <p>个人量化学习、ETF 相对强弱观察、特征研究和模型实验；不用于自动下单或资金管理。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-label">Technology Stack</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="doc-panel">
      <span class="stack-chip">Python 3.12</span>
      <span class="stack-chip">pandas</span>
      <span class="stack-chip">NumPy</span>
      <span class="stack-chip">LightGBM</span>
      <span class="stack-chip">scikit-learn</span>
      <span class="stack-chip">DuckDB</span>
      <span class="stack-chip">Parquet</span>
      <span class="stack-chip">Streamlit</span>
      <span class="stack-chip">Plotly</span>
      <span class="stack-chip">SF Fed News Sentiment</span>
      <span class="stack-chip">GPR Index</span>
      <span class="stack-chip">uv</span>
      <span class="stack-chip">pytest</span>
      <span class="stack-chip">Ruff</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Repository Guide</div>', unsafe_allow_html=True)
guide_left, guide_right = st.columns([1, 1])
with guide_left:
    st.markdown("### 核心目录")
    st.code(
        """trade/
├── app.py                    # 交互预测看板
├── pages/1_项目说明.py        # 当前技术文档页
├── configs/default.toml      # 离线配置（全部合成数据）
├── configs/free.toml         # 免费真实数据（Nasdaq / FRED / SF Fed / GPR）
├── configs/live.toml         # 真实数据源配置
├── src/etf_mvp/
│   ├── data/                 # 数据层（按数据种类划分）
│   │   ├── kinds.py          # 数据种类与 point-in-time 结构
│   │   ├── sources/          # market / macro / news / geopolitical / central_bank
│   │   ├── registry.py       # 各种类可选数据源
│   │   ├── store.py          # Parquet / DuckDB
│   │   └── service.py        # 采集与加载
│   ├── features/             # 每类数据一个因子模块，按可用日期对齐
│   ├── intraday.py           # QQQ / SOXX 独立5分钟数据、训练与走步回测
│   ├── modeling.py           # 模型训练与评估
│   └── pipeline.py           # 端到端流水线
├── tests/                    # 自动化测试
├── data/                     # 本地数据（Git 忽略）
└── artifacts/                # 预测与模型（Git 忽略）""",
        language="text",
    )
with guide_right:
    st.markdown("### 运行流程")
    st.code(
        """# 安装依赖
uv sync --extra dev

# 运行离线演示流水线
uv run etf-mvp run --provider synthetic

# 查看各类数据源
uv run etf-mvp sources

# 免费真实数据（仅 FRED 需要 API Key）
uv run etf-mvp --config configs/free.toml run

# QQQ / SOXX 多年5分钟数据、独立训练与独立回测
uv run etf-mvp --config configs/free.toml intraday-run --start 2018-01-01 --feed iex

# 只更新某一类数据，例如宏观
uv run etf-mvp --config configs/free.toml fetch --kind macro

# 启动前端
ETF_MVP_CONFIG=configs/free.toml uv run streamlit run app.py""",
        language="bash",
    )

st.markdown(
    """
    <div class="boundary-note">
      <strong>设计原则：</strong>数据层已接入行情、宏观、新闻情绪、地缘风险与美联储议息，所有非行情数据按“可用日期”对齐，并对周频发布的新闻/GPR指数采用保守滞后以避免前视偏差；次日 OHLC 模型采用五折扩展窗口样本外回测，QQQ 与 SOXX 另用多年5分钟数据分别训练独立序列模型并单独回测；不会在 MVP 阶段引入券商账户或自动交易权限。
    </div>
    """,
    unsafe_allow_html=True,
)
