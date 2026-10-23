from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from etf_mvp.config import load_config
from etf_mvp.data.kinds import DataKind
from etf_mvp.forecast import NextSessionForecast, forecast_next_session
from etf_mvp.interpretation import (
    DEEPSEEK_MODEL,
    ForecastInterpretation,
    build_forecast_context,
    deepseek_is_configured,
    interpret_forecast,
    load_local_env,
)
from etf_mvp.pipeline import run_pipeline, train_models

ETF_SHORT_NAMES = {
    "SPY": "标普500",
    "QQQ": "纳指100",
    "IWM": "罗素2000",
    "XLK": "科技板块",
    "XLF": "金融板块",
    "XLE": "能源板块",
    "TLT": "长期美债",
    "GLD": "黄金基金",
    "SOXX": "费城半导体",
}


def etf_display_name(symbol: str) -> str:
    short_name = ETF_SHORT_NAMES.get(symbol)
    return f"{symbol} · {short_name}" if short_name else symbol


st.set_page_config(
    page_title="ETF Research Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fb; }
    [data-testid="stSidebar"] { background: #101827; }
    [data-testid="stSidebar"] * { color: #f8fafc; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div {
        background-color: #ffffff;
        border-color: #cbd5e1;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] div {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] [data-baseweb="tag"],
    [data-testid="stSidebar"] [data-baseweb="tag"] * {
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        fill: #475569;
    }
    [data-testid="stSidebar"] [data-testid="stSelectbox"] .react-aria-ComboBox * {
        color: #0f172a !important;
        -webkit-text-fill-color: #0f172a !important;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
    }
    .hero {
        padding: 22px 26px;
        border-radius: 18px;
        color: white;
        background: linear-gradient(120deg, #111827, #1d4ed8);
        margin-bottom: 18px;
    }
    .hero h1 { margin: 0 0 6px 0; font-size: 2rem; }
    .hero p { margin: 0; color: #dbeafe; }
    .tag {
        display: inline-block;
        margin-top: 12px;
        padding: 4px 10px;
        border: 1px solid rgba(255,255,255,.25);
        border-radius: 999px;
        font-size: .8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_results(
    artifacts_dir_text: str, prices_path_text: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifacts_dir = Path(artifacts_dir_text)
    predictions_path = artifacts_dir / "predictions.csv"
    metrics_path = artifacts_dir / "metrics.json"
    importance_path = artifacts_dir / "feature_importance.csv"
    prices_path = Path(prices_path_text)

    predictions = (
        pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    )
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as metrics_file:
            metrics = pd.DataFrame(json.load(metrics_file))
    else:
        metrics = pd.DataFrame()
    importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
    prices = pd.read_parquet(prices_path) if prices_path.exists() else pd.DataFrame()
    if not prices.empty:
        prices["date"] = pd.to_datetime(prices["date"])
    return predictions, metrics, importance, prices


@st.cache_data(show_spinner=False)
def load_next_session_forecast(
    prices_path_text: str,
    features_path_text: str,
    symbol: str,
    random_seed: int,
    prices_mtime_ns: int,
    features_mtime_ns: int,
    forecast_version: int,
) -> NextSessionForecast:
    del prices_mtime_ns, features_mtime_ns, forecast_version
    prices_frame = pd.read_parquet(prices_path_text)
    feature_table = pd.read_parquet(features_path_text)
    return forecast_next_session(
        prices_frame,
        symbol,
        random_seed=random_seed,
        feature_table=feature_table,
    )


@st.cache_data(show_spinner=False, ttl=1800)
def load_llm_interpretation(
    context_json: str,
    model: str,
) -> ForecastInterpretation:
    return interpret_forecast(json.loads(context_json), model=model)


def format_feature_name(name: str) -> str:
    if "__" not in name:
        return name.replace("_", " ")
    symbol, metric = name.split("__", 1)
    special_labels = {
        "news__economic_sentiment__level": "新闻情绪 · 当前水平",
        "news__economic_sentiment__chg_5d": "新闻情绪 · 5日变化",
        "news__economic_sentiment__z_60d": "新闻情绪 · 60日标准分",
        "news__economic_sentiment__z_252d": "新闻情绪 · 252日标准分",
        "geo__gpr_level": "地缘风险 · GPR水平",
        "geo__gpr_chg_5d": "地缘风险 · 5日变化",
        "geo__gpr_chg_20d": "地缘风险 · 20日变化",
        "geo__gpr_z_60d": "地缘风险 · 60日标准分",
        "geo__gpr_z_252d": "地缘风险 · 252日标准分",
        "geo__threats_z_252d": "地缘威胁 · 252日标准分",
        "geo__acts_z_252d": "地缘行动 · 252日标准分",
    }
    if name in special_labels:
        return special_labels[name]
    labels = {
        "return_1d": "1日收益",
        "return_2d": "2日收益",
        "return_5d": "5日收益",
        "return_10d": "10日收益",
        "return_20d": "20日收益",
        "volatility_5d": "5日波动率",
        "volatility_10d": "10日波动率",
        "volatility_20d": "20日波动率",
        "close_to_sma_5": "距5日均线",
        "close_to_sma_10": "距10日均线",
        "close_to_sma_20": "距20日均线",
        "close_to_sma_50": "距50日均线",
        "volume_zscore_20d": "20日成交量强弱",
        "rsi_14": "RSI(14)",
        "downside_20d": "20日下行波动",
    }
    return f"{etf_display_name(symbol)} · {labels.get(metric, metric.replace('_', ' '))}"


def period_start(max_date: pd.Timestamp, period: str) -> pd.Timestamp | None:
    offsets = {
        "6个月": pd.DateOffset(months=6),
        "1年": pd.DateOffset(years=1),
        "3年": pd.DateOffset(years=3),
        "5年": pd.DateOffset(years=5),
    }
    return max_date - offsets[period] if period in offsets else None


config = load_config(os.getenv("ETF_MVP_CONFIG", "configs/default.toml"))

st.sidebar.title("研究控制台")
st.sidebar.caption("调整参数后可重新训练模型")
selected_targets = st.sidebar.multiselect(
    "预测标的",
    options=list(config.universe.symbols),
    default=list(config.universe.targets),
    max_selections=4,
    format_func=etf_display_name,
)
horizon_days = st.sidebar.select_slider(
    "预测周期（交易日）",
    options=[1, 3, 5, 10, 20],
    value=config.model.horizon_days,
)
probability_threshold = st.sidebar.slider(
    "上涨判断阈值",
    min_value=0.40,
    max_value=0.70,
    value=float(config.model.probability_threshold),
    step=0.01,
    help="上涨概率达到该值时，界面标记为 UP。",
)
run_mode = st.sidebar.selectbox(
    "数据模式",
    ["使用现有行情", "重新生成演示数据", "Nasdaq 免费真实日线"],
)
run_clicked = st.sidebar.button(
    "运行研究模型",
    type="primary",
    width="stretch",
    disabled=not selected_targets,
)

if run_clicked:
    runtime_config = replace(
        config,
        universe=replace(config.universe, targets=tuple(selected_targets)),
        model=replace(
            config.model,
            horizon_days=horizon_days,
            probability_threshold=probability_threshold,
        ),
    )
    try:
        with st.spinner("正在生成特征、训练模型并计算样本外指标……"):
            if run_mode == "使用现有行情":
                train_models(runtime_config)
            elif run_mode == "重新生成演示数据":
                run_pipeline(runtime_config, provider_name="synthetic")
            else:
                run_pipeline(runtime_config, provider_name="nasdaq")
        st.cache_data.clear()
        st.sidebar.success("运行完成，结果已更新")
    except Exception as error:
        st.sidebar.error(f"运行失败：{error}")

predictions, metrics, importance, prices = load_results(
    str(config.output.artifacts_dir), str(config.data.prices_path)
)

st.markdown(
    """
    <div class="hero">
      <h1>US ETF Research Lab</h1>
      <p>用可解释的量化模型观察美股 ETF 中短期方向概率。</p>
      <span class="tag">研究用途 · 不连接券商 · 不执行交易</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if predictions.empty:
    st.warning("还没有预测结果。请在左侧选择标的并点击“运行研究模型”。")
    st.stop()

display_predictions = predictions.copy()
display_predictions["symbol_label"] = display_predictions["symbol"].map(
    etf_display_name
)
display_predictions["signal"] = display_predictions["probability_up"].map(
    lambda probability: "UP" if probability >= probability_threshold else "DOWN"
)

latest_date = display_predictions["prediction_date"].max()
model_name = ", ".join(sorted(display_predictions["model"].unique()))
st.caption(
    f"数据截止：{latest_date}　|　模型：{model_name}　|　"
    f"当前阈值：{probability_threshold:.0%}"
)

overview_tab, market_tab, model_tab, guide_tab = st.tabs(
    ["预测概览", "行情分析", "模型评估", "使用说明"]
)

with overview_tab:
    metric_columns = st.columns(len(display_predictions))
    for column, row in zip(
        metric_columns, display_predictions.itertuples(index=False), strict=True
    ):
        direction_text = "看涨" if row.signal == "UP" else "谨慎"
        column.metric(
            label=f"{etf_display_name(row.symbol)} · 未来 {row.horizon_days} 日",
            value=f"{row.probability_up:.1%}",
            delta=direction_text,
            delta_color="normal" if row.signal == "UP" else "inverse",
        )

    chart_left, chart_right = st.columns([1.35, 1])
    with chart_left:
        st.subheader("上涨概率对比")
        probability_figure = px.bar(
            display_predictions,
            x="symbol_label",
            y="probability_up",
            color="signal",
            text=display_predictions["probability_up"].map(lambda value: f"{value:.1%}"),
            range_y=[0, 1],
            color_discrete_map={"UP": "#16a34a", "DOWN": "#ef4444"},
        )
        probability_figure.add_hline(
            y=probability_threshold,
            line_dash="dash",
            line_color="#64748b",
            annotation_text="判断阈值",
        )
        probability_figure.update_layout(
            xaxis_title=None,
            yaxis_title="上涨概率",
            yaxis_tickformat=".0%",
            legend_title=None,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(probability_figure, width="stretch")

    with chart_right:
        st.subheader("如何解读")
        strongest = display_predictions.loc[display_predictions["probability_up"].idxmax()]
        weakest = display_predictions.loc[display_predictions["probability_up"].idxmin()]
        st.write(
            f"**相对最强：** {etf_display_name(strongest['symbol'])}，"
            f"上涨概率 {strongest['probability_up']:.1%}"
        )
        st.write(
            f"**相对最弱：** {etf_display_name(weakest['symbol'])}，"
            f"上涨概率 {weakest['probability_up']:.1%}"
        )
        st.write(f"**模型周期：** 未来 {int(strongest['horizon_days'])} 个交易日")
        st.info("概率接近 50% 表示模型缺少明确方向，不应解读为交易信号。")

with market_tab:
    if prices.empty:
        st.warning("未找到行情数据。")
    else:
        available_symbols = sorted(prices["symbol"].unique())
        control_left, control_right = st.columns([1, 2])
        selected_symbol = control_left.selectbox(
            "查看 ETF",
            available_symbols,
            format_func=etf_display_name,
        )
        selected_period = control_right.radio(
            "时间范围",
            ["6个月", "1年", "3年", "5年", "全部"],
            index=1,
            horizontal=True,
        )
        symbol_prices = prices.loc[prices["symbol"] == selected_symbol].sort_values("date")
        start_date = period_start(symbol_prices["date"].max(), selected_period)
        if start_date is not None:
            symbol_prices = symbol_prices.loc[symbol_prices["date"] >= start_date]

        candle_figure = go.Figure(
            data=[
                go.Candlestick(
                    x=symbol_prices["date"],
                    open=symbol_prices["open"],
                    high=symbol_prices["high"],
                    low=symbol_prices["low"],
                    close=symbol_prices["close"],
                    increasing_line_color="#16a34a",
                    decreasing_line_color="#ef4444",
                    name=etf_display_name(selected_symbol),
                )
            ]
        )
        candle_figure.update_layout(
            title=f"{etf_display_name(selected_symbol)} 日线走势",
            xaxis_rangeslider_visible=False,
            yaxis_title="价格",
            height=470,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(candle_figure, width="stretch")

        st.divider()
        forecast_header, forecast_action = st.columns([3, 1])
        with forecast_header:
            st.subheader("下一交易日行情预测")
            st.caption("用历史日线特征预测次日 OHLC，并生成 5 分钟分时情景路径。")
        forecast_key = (
            f"{selected_symbol}:{prices['date'].max().date().isoformat()}"
        )
        forecast_clicked = forecast_action.button(
            f"生成 {selected_symbol} 预测",
            type="primary",
            width="stretch",
            key="run_next_session_forecast",
        )
        if forecast_clicked:
            st.session_state["next_session_forecast_key"] = forecast_key

        if st.session_state.get("next_session_forecast_key") == forecast_key:
            try:
                with st.spinner(f"正在训练 {selected_symbol} 次日 OHLC 模型……"):
                    prices_mtime_ns = config.data.prices_path.stat().st_mtime_ns
                    features_mtime_ns = config.data.features_path.stat().st_mtime_ns
                    forecast = load_next_session_forecast(
                        str(config.data.prices_path),
                        str(config.data.features_path),
                        selected_symbol,
                        config.model.random_seed,
                        prices_mtime_ns,
                        features_mtime_ns,
                        5,
                    )

                forecast_columns = st.columns(5)
                forecast_values = [
                    ("预测开盘", forecast.predicted_open),
                    ("预测最高", forecast.predicted_high),
                    ("预测最低", forecast.predicted_low),
                    ("预测收盘", forecast.predicted_close),
                ]
                for column, (label, value) in zip(
                    forecast_columns[:4], forecast_values, strict=True
                ):
                    column.metric(
                        label,
                        f"${value:,.2f}",
                        f"{value / forecast.last_close - 1:+.2%}",
                    )
                forecast_columns[4].metric(
                    "预测涨跌幅",
                    f"{forecast.expected_return:+.2%}",
                    f"前收 ${forecast.last_close:,.2f}",
                )

                st.caption(
                    f"数据截止 {forecast.as_of_date.date()} · "
                    f"预测日期 {forecast.forecast_date.date()} · "
                    f"模型 {forecast.model_name} · {forecast.feature_count} 个因子"
                )

                intraday_display = forecast.intraday.copy()
                intraday_display["average"] = intraday_display["close"].expanding().mean()
                intraday_display["change"] = (
                    intraday_display["close"] / forecast.last_close - 1
                )
                intraday_figure = go.Figure()
                intraday_figure.add_trace(
                    go.Scatter(
                        x=intraday_display["timestamp"],
                        y=intraday_display["close"],
                        customdata=intraday_display["change"],
                        mode="lines",
                        line=dict(color="#1683ff", width=2.5),
                        fill="tozeroy",
                        fillcolor="rgba(22, 131, 255, 0.10)",
                        name="预测价格",
                        hovertemplate=(
                            "%{x|%H:%M}<br>预测价 $%{y:.2f}<br>"
                            "较前收 %{customdata:+.2%}<extra></extra>"
                        ),
                    )
                )
                intraday_figure.add_trace(
                    go.Scatter(
                        x=intraday_display["timestamp"],
                        y=intraday_display["average"],
                        mode="lines",
                        line=dict(color="#f59e0b", width=1.8),
                        name="预测均价",
                        hovertemplate="%{x|%H:%M}<br>均价 $%{y:.2f}<extra></extra>",
                    )
                )
                turning_points = pd.DataFrame(
                    {
                        "timestamp": [
                            intraday_display.loc[
                                intraday_display["high"].idxmax(), "timestamp"
                            ],
                            intraday_display.loc[
                                intraday_display["low"].idxmin(), "timestamp"
                            ],
                        ],
                        "price": [
                            intraday_display["high"].max(),
                            intraday_display["low"].min(),
                        ],
                    }
                )
                intraday_figure.add_trace(
                    go.Scatter(
                        x=turning_points["timestamp"],
                        y=turning_points["price"],
                        mode="markers+text",
                        marker=dict(color="#1683ff", size=7),
                        text=turning_points["price"].map(lambda value: f"${value:.2f}"),
                        textposition=["top center", "bottom center"],
                        name="预测高低点",
                        hoverinfo="skip",
                    )
                )
                intraday_figure.add_trace(
                    go.Scatter(
                        x=intraday_display["timestamp"],
                        y=intraday_display["change"],
                        yaxis="y2",
                        mode="lines",
                        line=dict(color="rgba(0, 0, 0, 0)"),
                        showlegend=False,
                        hoverinfo="skip",
                        name="较前收涨跌幅",
                    )
                )
                intraday_figure.add_hline(
                    y=forecast.last_close,
                    line_dash="dash",
                    line_color="#ef4444",
                    opacity=0.65,
                    annotation_text=f"前收 ${forecast.last_close:.2f}",
                    annotation_position="bottom right",
                )
                price_span = max(forecast.predicted_high - forecast.predicted_low, 0.01)
                y_min = forecast.predicted_low - price_span * 0.12
                y_max = forecast.predicted_high + price_span * 0.12
                change_min = y_min / forecast.last_close - 1
                change_max = y_max / forecast.last_close - 1
                percent_ticks = [
                    change_min + (change_max - change_min) * index / 4
                    for index in range(5)
                ]
                intraday_figure.update_layout(
                    title=(
                        f"{etf_display_name(selected_symbol)} · "
                        f"{forecast.forecast_date.date()} "
                        "预测分时走势"
                    ),
                    xaxis=dict(
                        title="美东时间",
                        tickformat="%H:%M",
                        dtick=60 * 60 * 1000,
                        showgrid=True,
                        gridcolor="#eef2f7",
                    ),
                    yaxis=dict(
                        title="预测价格",
                        range=[y_min, y_max],
                        tickprefix="$",
                        showgrid=True,
                        gridcolor="#e5e7eb",
                    ),
                    yaxis2=dict(
                        title="较前收涨跌幅",
                        overlaying="y",
                        side="right",
                        range=[change_min, change_max],
                        tickvals=percent_ticks,
                        ticktext=[f"{value:+.2%}" for value in percent_ticks],
                        showgrid=False,
                        zeroline=False,
                    ),
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.12, x=0),
                    height=460,
                    margin=dict(l=10, r=65, t=70, b=10),
                )
                st.plotly_chart(intraday_figure, width="stretch")

                st.markdown("#### 扩展窗口样本外回测")
                st.caption(
                    "取历史末段 20% 样本并按时间拆成 5 折；每折只使用该折开始日期之前的"
                    "数据训练，再预测后续交易日，避免随机切分和未来数据泄漏。"
                )
                quality_columns = st.columns(4)
                quality_columns[0].metric(
                    "方向准确率", f"{forecast.direction_accuracy:.1%}"
                )
                quality_columns[1].metric(
                    "开盘收益 MAE", f"{forecast.open_return_mae:.2%}"
                )
                quality_columns[2].metric(
                    "高/低收益 MAE",
                    f"{forecast.high_return_mae:.2%} / {forecast.low_return_mae:.2%}",
                )
                quality_columns[3].metric(
                    "收盘收益 MAE", f"{forecast.close_return_mae:.2%}"
                )

                backtest_display = forecast.backtest.tail(160)
                backtest_figure = go.Figure()
                backtest_figure.add_trace(
                    go.Scatter(
                        x=backtest_display["forecast_date"],
                        y=backtest_display["actual_close"],
                        mode="lines",
                        line=dict(color="#111827", width=2),
                        name="实际收盘",
                    )
                )
                backtest_figure.add_trace(
                    go.Scatter(
                        x=backtest_display["forecast_date"],
                        y=backtest_display["predicted_close"],
                        mode="lines",
                        line=dict(color="#1683ff", width=1.8, dash="dot"),
                        name="样本外预测收盘",
                    )
                )
                backtest_figure.update_layout(
                    xaxis_title="预测交易日",
                    yaxis_title="收盘价",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.12, x=0),
                    height=350,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                st.plotly_chart(backtest_figure, width="stretch")
                st.caption(
                    f"回测样本 {len(forecast.backtest)} 个交易日 · "
                    f"{forecast.backtest['forecast_date'].min().date()} 至 "
                    f"{forecast.backtest['forecast_date'].max().date()} · "
                    "图中展示最近 160 个样本外预测。"
                )
                st.download_button(
                    "下载滚动回测 CSV",
                    data=forecast.backtest.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_symbol}_walk_forward_backtest.csv",
                    mime="text/csv",
                )

                st.markdown("#### 可解释化预测")
                st.caption(
                    "影响占比是 LLM 对技术、宏观、地缘与新闻的情境性解释，"
                    "不是模型因果贡献或 SHAP 值；卡片内的局部影响来自收盘收益模型的"
                    "单次预测贡献，新闻与地缘因子已直接参与模型计算。"
                )
                load_local_env()
                deepseek_model = os.getenv("DEEPSEEK_MODEL", DEEPSEEK_MODEL)
                interpretation_key = (
                    f"{forecast_key}:{forecast.predicted_close:.6f}:{deepseek_model}"
                )
                if not deepseek_is_configured():
                    st.warning("尚未配置 DEEPSEEK_API_KEY，无法生成 LLM 解读。")
                else:
                    interpret_clicked = st.button(
                        "生成 LLM 解读",
                        key="run_llm_interpretation",
                        width="stretch",
                    )
                    if interpret_clicked:
                        st.session_state["llm_interpretation_key"] = interpretation_key

                    if (
                        st.session_state.get("llm_interpretation_key")
                        == interpretation_key
                    ):
                        context = build_forecast_context(
                            forecast,
                            prices,
                            data_provider=config.data.provider,
                        )
                        try:
                            with st.spinner(
                                "DeepSeek 正在检索并分析技术、宏观、地缘与短期新闻……"
                            ):
                                interpretation = load_llm_interpretation(
                                    json.dumps(
                                        context,
                                        ensure_ascii=False,
                                        sort_keys=True,
                                    ),
                                    deepseek_model,
                                )

                            st.info(interpretation.summary)
                            confidence_left, confidence_right = st.columns(2)
                            confidence_left.metric(
                                "LLM 解读置信度", f"{interpretation.confidence}%"
                            )
                            confidence_left.caption(
                                "口径：由证据完整性、一致性与时效性综合评估，"
                                "不是行情上涨概率。"
                            )
                            confidence_right.metric(
                                "归因占比合计",
                                f"{sum(item.attribution_pct for item in interpretation.sections)}%",
                            )
                            confidence_right.caption(
                                "公式：技术形态 + 宏观面 + 地缘因素 + 短期新闻，"
                                "归一化后合计为 100%。"
                            )

                            for row_start in (0, 2):
                                section_columns = st.columns(2)
                                for column, section in zip(
                                    section_columns,
                                    interpretation.sections[row_start : row_start + 2],
                                    strict=True,
                                ):
                                    with column.container(border=True):
                                        st.markdown(
                                            f"**{section.title} · {section.direction}**"
                                        )
                                        st.progress(
                                            section.attribution_pct / 100,
                                            text=(
                                                f"影响占比 {section.attribution_pct}% · "
                                                f"影响分 {section.impact_score:+d}"
                                            ),
                                        )
                                        st.write(section.analysis)
                                        if section.model_factors:
                                            st.markdown("**对应模型特征**")
                                            for factor in section.model_factors:
                                                share_text = (
                                                    f" · 局部影响 {factor.local_share:.1%}"
                                                    if factor.local_share is not None
                                                    else ""
                                                )
                                                st.markdown(
                                                    f"- {format_feature_name(factor.feature)}"
                                                    f" · {factor.direction}{share_text}"
                                                )
                                        else:
                                            st.caption("未匹配到直接模型特征。")
                                        if section.context_factors:
                                            st.markdown("**外部情境因子**")
                                            st.write("、".join(section.context_factors))
                                        for evidence in section.evidence:
                                            st.markdown(f"- {evidence}")

                            if interpretation.risk_notes:
                                st.markdown("**风险与不确定性**")
                                for risk_note in interpretation.risk_notes:
                                    st.markdown(f"- {risk_note}")
                            if interpretation.sources:
                                st.markdown("**参考来源**")
                                for source in interpretation.sources:
                                    st.markdown(f"- [{source.title}]({source.url})")
                            usage_text = (
                                f" · {interpretation.total_tokens:,} tokens"
                                if interpretation.total_tokens is not None
                                else ""
                            )
                            st.caption(
                                f"由 {interpretation.model} 生成{usage_text} · 仅供研究"
                            )
                        except Exception as error:
                            st.error(f"LLM 解读生成失败：{error}")
                    else:
                        st.info("点击“生成 LLM 解读”后调用 DeepSeek；结果缓存 30 分钟。")

                interval_left, interval_right = st.columns(2)
                interval_left.metric(
                    "预测收盘区间",
                    f"${forecast.close_lower:,.2f} – ${forecast.close_upper:,.2f}",
                )
                interval_right.metric(
                    "回测覆盖区间",
                    f"{forecast.backtest['forecast_date'].min().date()} – "
                    f"{forecast.backtest['forecast_date'].max().date()}",
                )
                st.warning(
                    "分时折线是由预测 OHLC 与近 20 日波动率约束生成的情景路径，"
                    "不是分钟级模型预测，也没有分钟级回测；上方回测仅验证次日 OHLC 模型。"
                    "预测日期按工作日估算，暂未处理美股节假日。"
                )
                st.download_button(
                    "下载预测分时数据 CSV",
                    data=forecast.intraday.to_csv(index=False).encode("utf-8"),
                    file_name=(
                        f"{selected_symbol}_{forecast.forecast_date.date()}_forecast.csv"
                    ),
                    mime="text/csv",
                )
            except Exception as error:
                st.error(f"行情预测生成失败：{error}")
        else:
            st.info("点击“生成预测”后，页面会训练次日 OHLC 模型并绘制分时折线。")

        st.divider()
        compare_symbols = st.multiselect(
            "归一化走势对比",
            available_symbols,
            default=available_symbols[:3],
            max_selections=5,
            format_func=etf_display_name,
        )
        if compare_symbols:
            comparison = (
                prices.loc[prices["symbol"].isin(compare_symbols)]
                .pivot(index="date", columns="symbol", values="close")
                .sort_index()
                .ffill()
            )
            comparison_start = period_start(comparison.index.max(), selected_period)
            if comparison_start is not None:
                comparison = comparison.loc[comparison.index >= comparison_start]
            normalized = comparison / comparison.iloc[0] * 100
            normalized_figure = px.line(
                normalized,
                labels={"value": "起点 = 100", "date": "日期", "variable": "ETF"},
            )
            normalized_figure.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(normalized_figure, width="stretch")

with model_tab:
    if metrics.empty:
        st.warning("未找到模型评估数据。")
    else:
        st.subheader("样本外指标")
        display_metrics = metrics[
            [
                "symbol",
                "accuracy",
                "roc_auc",
                "brier_score",
                "positive_rate",
                "test_start",
                "test_end",
            ]
        ].copy()
        for column in ["accuracy", "roc_auc", "positive_rate"]:
            display_metrics[column] = display_metrics[column].map(
                lambda value: "—" if pd.isna(value) else f"{value:.1%}"
            )
        display_metrics["brier_score"] = display_metrics["brier_score"].map(
            lambda value: f"{value:.3f}"
        )
        display_metrics.columns = [
            "ETF",
            "准确率",
            "ROC AUC",
            "Brier",
            "上涨基准率",
            "测试开始",
            "测试结束",
        ]
        st.dataframe(display_metrics, hide_index=True, width="stretch")

    if not importance.empty:
        st.subheader("特征重要性")
        importance_symbol = st.selectbox(
            "解释目标",
            sorted(importance["symbol"].unique()),
            key="importance_symbol",
            format_func=etf_display_name,
        )
        selected_importance = (
            importance.loc[importance["symbol"] == importance_symbol]
            .nlargest(15, "importance")
            .sort_values("importance")
            .copy()
        )
        selected_importance["feature_label"] = selected_importance["feature"].map(
            format_feature_name
        )
        importance_figure = px.bar(
            selected_importance,
            x="importance",
            y="feature_label",
            orientation="h",
            color="importance",
            color_continuous_scale="Blues",
        )
        importance_figure.update_layout(
            yaxis_title=None,
            xaxis_title="重要性",
            coloraxis_showscale=False,
            height=500,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(importance_figure, width="stretch")

        by_kind_path = config.output.artifacts_dir / "importance_by_kind.csv"
        if by_kind_path.exists():
            st.subheader("各类数据的贡献")
            kind_labels = {kind.value: kind.label for kind in DataKind}
            by_kind = pd.read_csv(by_kind_path)
            by_kind = by_kind.loc[by_kind["symbol"] == importance_symbol].copy()
            by_kind["数据类别"] = by_kind["kind"].map(kind_labels).fillna(by_kind["kind"])
            by_kind["占比"] = by_kind["importance"] / by_kind["importance"].sum()
            kind_figure = px.bar(
                by_kind.sort_values("占比"),
                x="占比",
                y="数据类别",
                orientation="h",
                text=by_kind.sort_values("占比")["占比"].map(lambda value: f"{value:.0%}"),
            )
            kind_figure.update_layout(
                xaxis_tickformat=".0%",
                yaxis_title=None,
                height=300,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(kind_figure, width="stretch")

with guide_tab:
    st.subheader("三步使用")
    st.markdown(
        """
        1. 在左侧选择要研究的 ETF、预测周期和概率阈值。
        2. 选择现有行情、演示数据或 Nasdaq，然后点击 **运行研究模型**。
        3. 在各标签页查看概率、历史走势、样本外指标和特征重要性。

        **指标提示**
        - `ROC AUC` 越接近 1，模型排序能力越强；约 0.5 表示接近随机。
        - `Brier` 越低，概率预测的校准误差越小。
        - 单次概率不能代表确定收益，应结合长期样本外表现判断。
        """
    )
    st.warning("本项目仅用于模型研究和学习，不构成投资建议，也不会发送任何交易指令。")
