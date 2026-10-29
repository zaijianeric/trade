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
from etf_mvp.forecast import (
    NextSessionForecast,
    compare_backtests_on_common_dates,
    forecast_next_session,
)
from etf_mvp.interpretation import (
    DEEPSEEK_MODEL,
    ForecastInterpretation,
    build_forecast_context,
    deepseek_is_configured,
    interpret_forecast,
    load_local_env,
)
from etf_mvp.intraday import INTRADAY_SLOTS, INTRADAY_SYMBOLS, roll_intraday_forecast
from etf_mvp.pipeline import DAILY_FORECAST_HORIZONS, fetch_data, train_models

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
    initial_sidebar_state="collapsed",
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
    artifacts_dir_text: str,
    prices_path_text: str,
    predictions_mtime_ns: int,
    metrics_mtime_ns: int,
    importance_mtime_ns: int,
    prices_mtime_ns: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del predictions_mtime_ns, metrics_mtime_ns, importance_mtime_ns, prices_mtime_ns
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
def load_intraday_results(
    artifact_dir_text: str,
    forecast_mtime_ns: int,
    metrics_mtime_ns: int,
    backtest_mtime_ns: int,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    del forecast_mtime_ns, metrics_mtime_ns, backtest_mtime_ns
    artifact_dir = Path(artifact_dir_text)
    forecast = pd.read_csv(artifact_dir / "forecast.csv", parse_dates=["timestamp"])
    backtest = pd.read_csv(
        artifact_dir / "backtest_sessions.csv", parse_dates=["session_date"]
    )
    with (artifact_dir / "metrics.json").open(encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    return forecast, metrics, backtest


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
probability_threshold = float(config.model.probability_threshold)

predictions, metrics, importance, prices = load_results(
    str(config.output.artifacts_dir),
    str(config.data.prices_path),
    (config.output.artifacts_dir / "predictions.csv").stat().st_mtime_ns,
    (config.output.artifacts_dir / "metrics.json").stat().st_mtime_ns,
    (config.output.artifacts_dir / "feature_importance.csv").stat().st_mtime_ns,
    config.data.prices_path.stat().st_mtime_ns,
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

with st.expander("模型配置", expanded=False):
    st.caption("每次训练固定生成未来 1、3、5 个交易日的预测，最多两项并行执行。")
    with st.form("daily_model_config"):
        config_left, config_right = st.columns(2)
        selected_targets = config_left.multiselect(
            "预测标的",
            options=list(config.universe.symbols),
            default=list(config.universe.targets),
            max_selections=4,
            format_func=etf_display_name,
        )
        probability_threshold = config_right.slider(
            "上涨判断阈值",
            min_value=0.40,
            max_value=0.70,
            value=probability_threshold,
            step=0.01,
            help="上涨概率达到该值时，界面标记为 UP。",
        )
        run_mode = config_left.selectbox(
            "数据模式",
            ["使用现有行情", "重新拉取 Alpaca 日线", "重新生成演示数据"],
        )
        run_clicked = st.form_submit_button(
            "并行训练 1 / 3 / 5 日模型",
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
            probability_threshold=probability_threshold,
        ),
    )
    try:
        with st.spinner("正在并行训练 1 / 3 / 5 日模型并计算样本外指标……"):
            if run_mode == "使用现有行情":
                bundle = None
            elif run_mode == "重新拉取 Alpaca 日线":
                fetch_data(runtime_config, [DataKind.MARKET], provider_name="alpaca")
                bundle = None
            else:
                bundle = fetch_data(runtime_config, provider_name="synthetic")
            train_models(runtime_config, bundle, horizons=DAILY_FORECAST_HORIZONS)
        st.cache_data.clear()
        st.success("训练完成：1 / 3 / 5 日结果已更新。")
        st.rerun()
    except Exception as error:
        st.error(f"运行失败：{error}")

if predictions.empty:
    st.warning("还没有预测结果。请在下方展开“模型配置”并运行训练。")
    st.stop()

display_predictions = predictions.copy()
display_predictions["symbol_label"] = display_predictions["symbol"].map(
    etf_display_name
)
display_predictions["signal"] = display_predictions["probability_up"].map(
    lambda probability: "UP" if probability >= probability_threshold else "DOWN"
)
display_predictions["horizon_label"] = display_predictions["horizon_days"].map(
    lambda horizon: f"{int(horizon)} 日"
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
    st.subheader("1 / 3 / 5 日预测")
    for symbol in display_predictions["symbol"].drop_duplicates():
        symbol_predictions = display_predictions.loc[
            display_predictions["symbol"] == symbol
        ].set_index("horizon_days")
        with st.container(border=True):
            st.markdown(f"**{etf_display_name(symbol)}**")
            metric_columns = st.columns(len(DAILY_FORECAST_HORIZONS))
            for column, horizon_days in zip(
                metric_columns, DAILY_FORECAST_HORIZONS, strict=True
            ):
                if horizon_days not in symbol_predictions.index:
                    column.metric(f"未来 {horizon_days} 日", "--")
                    continue
                row = symbol_predictions.loc[horizon_days]
                direction_text = "看涨" if row.signal == "UP" else "谨慎"
                column.metric(
                    label=f"未来 {horizon_days} 日",
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
            color="horizon_label",
            barmode="group",
            text=display_predictions["probability_up"].map(lambda value: f"{value:.1%}"),
            range_y=[0, 1],
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
            legend_title="预测周期",
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(probability_figure, width="stretch")

    with chart_right:
        st.subheader("如何解读")
        for horizon_days in DAILY_FORECAST_HORIZONS:
            horizon_predictions = display_predictions.loc[
                display_predictions["horizon_days"] == horizon_days
            ]
            if horizon_predictions.empty:
                continue
            strongest = horizon_predictions.loc[horizon_predictions["probability_up"].idxmax()]
            weakest = horizon_predictions.loc[horizon_predictions["probability_up"].idxmin()]
            st.write(
                f"**{horizon_days} 日：** 最强 {etf_display_name(strongest['symbol'])} "
                f"{strongest['probability_up']:.1%}；最弱 "
                f"{etf_display_name(weakest['symbol'])} {weakest['probability_up']:.1%}"
            )
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

        minute_forecast = pd.DataFrame()
        minute_metrics: dict[str, object] = {}
        minute_backtest = pd.DataFrame()
        intraday_artifact_dir = config.output.artifacts_dir / "intraday" / selected_symbol
        intraday_forecast_path = intraday_artifact_dir / "forecast.csv"
        intraday_metrics_path = intraday_artifact_dir / "metrics.json"
        intraday_backtest_path = intraday_artifact_dir / "backtest_sessions.csv"
        intraday_live_status_path = intraday_artifact_dir / "live_status.json"
        minute_model_available = selected_symbol in INTRADAY_SYMBOLS and all(
            path.exists()
            for path in (
                intraday_forecast_path,
                intraday_metrics_path,
                intraday_backtest_path,
            )
        )
        if minute_model_available:
            # Keep the premarket forecast as the base for every re-roll. The live
            # forecast artifact is already anchored to the latest bars; feeding it
            # back into roll_intraday_forecast would ignore older cutoff choices.
            minute_forecast, minute_metrics, minute_backtest = load_intraday_results(
                str(intraday_artifact_dir),
                intraday_forecast_path.stat().st_mtime_ns,
                intraday_metrics_path.stat().st_mtime_ns,
                intraday_backtest_path.stat().st_mtime_ns,
            )

        rolling_bars = pd.DataFrame()
        actual_intraday = pd.DataFrame()
        rolling_cutoff = -1
        rolling_options = [-1, *range(INTRADAY_SLOTS)] if minute_model_available else []
        if minute_model_available and "predicted_price_q10" in minute_forecast:
            project_root = Path(config.data.prices_path).parent.parent.parent
            forecast_date_text = str(minute_metrics["forecast_date"])
            evaluation_path = (
                project_root
                / "data"
                / "intraday"
                / "evaluation"
                / f"{selected_symbol}_{forecast_date_text}_5min.parquet"
            )
            if evaluation_path.exists():
                rolling_bars = pd.read_parquet(evaluation_path).sort_values("slot")
                if selected_symbol == "SOXX":
                    actual_intraday = rolling_bars[["timestamp", "slot", "close"]].dropna()
                    actual_intraday["timestamp"] = pd.to_datetime(
                        actual_intraday["timestamp"], utc=True
                    ).dt.tz_convert("America/New_York")
        latest_observed_slot = (
            int(rolling_bars["slot"].max()) if not rolling_bars.empty else -1
        )

        st.divider()
        forecast_header, forecast_action = st.columns([3, 1])
        with forecast_header:
            st.subheader("下一交易日行情预测")
            if minute_model_available:
                st.caption(
                    "使用真实 5 分钟序列模型作为主预测；原日线 OHLC 模型作为基准，"
                    "在共同样本外交易日比较准确率。"
                )
                if intraday_live_status_path.exists():
                    with intraday_live_status_path.open(encoding="utf-8") as status_file:
                        live_status = json.load(status_file)
                    if live_status.get("session_date") == str(minute_metrics["forecast_date"]):
                        st.caption(
                            "轮询最新数据：截至 "
                            f"{live_status.get('cutoff_time', '-')}（"
                            f"{live_status.get('updated_at', '')}）。"
                            "拖动下方截止时点可回看更早的滚动结果。"
                        )
            else:
                st.caption("用历史日线特征预测次日 OHLC，并生成 5 分钟分时情景路径。")
        if rolling_options:
            rolling_cutoff = st.select_slider(
                "盘中滚动预测截止",
                options=rolling_options,
                value=latest_observed_slot,
                format_func=lambda slot: (
                    "盘前（不使用当日数据）"
                    if slot < 0
                    else (
                        pd.Timestamp("09:30") + pd.Timedelta(minutes=5 * int(slot))
                    ).strftime("%H:%M")
                ),
                help="选择时点后，仅用该时点及之前的真实5分钟行情锚定剩余预测。",
            )
            if rolling_bars.empty:
                st.caption("尚未获取当日实时分钟线；当前显示盘前基准预测。")
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
                with st.spinner(f"正在训练 {selected_symbol} 日线基准模型……"):
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

                if minute_model_available:
                    display_minute_forecast = minute_forecast.copy()
                    if rolling_cutoff >= 0:
                        display_minute_forecast = roll_intraday_forecast(
                            minute_forecast,
                            rolling_bars.loc[rolling_bars["slot"] <= rolling_cutoff],
                            float(minute_metrics["last_close"]),
                        )
                    intraday_display = display_minute_forecast.rename(
                        columns={"predicted_price": "close"}
                    ).copy()
                    display_last_close = float(minute_metrics["last_close"])
                    display_open = float(intraday_display["close"].iloc[0])
                    display_high = float(intraday_display["close"].max())
                    display_low = float(intraday_display["close"].min())
                    display_close = float(intraday_display["close"].iloc[-1])
                    display_forecast_date = pd.Timestamp(
                        str(minute_metrics["forecast_date"])
                    )
                    display_model = str(minute_metrics["model"])
                    display_feature_count = int(minute_metrics["feature_count"])
                else:
                    intraday_display = forecast.intraday.copy()
                    display_last_close = forecast.last_close
                    display_open = forecast.predicted_open
                    display_high = forecast.predicted_high
                    display_low = forecast.predicted_low
                    display_close = forecast.predicted_close
                    display_forecast_date = forecast.forecast_date
                    display_model = forecast.model_name
                    display_feature_count = forecast.feature_count

                if actual_intraday.empty or rolling_cutoff < 0:
                    display_actual_intraday = actual_intraday.iloc[0:0]
                else:
                    display_actual_intraday = actual_intraday.loc[
                        actual_intraday["slot"] <= rolling_cutoff
                    ]

                forecast_columns = st.columns(5)
                forecast_values = [
                    (
                        "预测首时点" if minute_model_available else "预测开盘",
                        display_open,
                    ),
                    ("预测最高", display_high),
                    ("预测最低", display_low),
                    ("预测收盘", display_close),
                ]
                for column, (label, value) in zip(
                    forecast_columns[:4], forecast_values, strict=True
                ):
                    column.metric(
                        label,
                        f"${value:,.2f}",
                        f"{value / display_last_close - 1:+.2%}",
                    )
                forecast_columns[4].metric(
                    "预测涨跌幅",
                    f"{display_close / display_last_close - 1:+.2%}",
                    f"前收 ${display_last_close:,.2f}",
                )

                st.caption(
                    f"预测日期 {display_forecast_date.date()} · "
                    f"主模型 {display_model} · {display_feature_count} 个分钟序列特征"
                    if minute_model_available
                    else (
                        f"数据截止 {forecast.as_of_date.date()} · "
                        f"预测日期 {forecast.forecast_date.date()} · "
                        f"模型 {forecast.model_name} · {forecast.feature_count} 个因子"
                    )
                )

                if minute_model_available and "predicted_return_q10" in intraday_display:
                    close_q10 = float(intraday_display["predicted_return_q10"].iloc[-1])
                    close_q50 = float(intraday_display["predicted_return"].iloc[-1])
                    close_q90 = float(intraday_display["predicted_return_q90"].iloc[-1])
                    if close_q10 >= 0:
                        probability_up = 0.95
                    elif close_q90 <= 0:
                        probability_up = 0.05
                    elif close_q50 >= 0:
                        probability_up = 1 - (
                            0.1 + 0.4 * (0 - close_q10) / max(close_q50 - close_q10, 1e-9)
                        )
                    else:
                        probability_up = 1 - (
                            0.5 + 0.4 * (0 - close_q50) / max(close_q90 - close_q50, 1e-9)
                        )
                    probability_columns = st.columns(2)
                    probability_columns[0].metric("收盘上涨概率参照", f"{probability_up:.1%}")
                    forecast_mode = (
                        str(display_minute_forecast["forecast_mode"].iloc[-1])
                        if "forecast_mode" in display_minute_forecast
                        else "premarket"
                    )
                    probability_columns[1].metric(
                        "预测模式",
                        "盘中滚动" if forecast_mode == "rolling" else "盘前预测",
                    )
                    st.caption(
                        "概率仅用于衡量分时趋势离不确定区间有多远；主输出仍是蓝色分时趋势。"
                    )

                intraday_display["average"] = intraday_display["close"].expanding().mean()
                intraday_display["change"] = (
                    intraday_display["close"] / display_last_close - 1
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
                        name=("5分钟模型预测" if minute_model_available else "预测价格"),
                        hovertemplate=(
                            "%{x|%H:%M}<br>预测价 $%{y:.2f}<br>"
                            "较前收 %{customdata:+.2%}<extra></extra>"
                        ),
                    )
                )
                if not display_actual_intraday.empty:
                    actual_change = display_actual_intraday["close"] / display_last_close - 1
                    intraday_figure.add_trace(
                        go.Scatter(
                            x=display_actual_intraday["timestamp"],
                            y=display_actual_intraday["close"],
                            customdata=actual_change,
                            mode="lines",
                            line=dict(color="#111827", width=2.2),
                            name=f"{selected_symbol} 真实5分钟行情",
                            hovertemplate=(
                                "%{x|%H:%M}<br>实际价 $%{y:.2f}<br>"
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
                                intraday_display["close"].idxmax(), "timestamp"
                            ],
                            intraday_display.loc[
                                intraday_display["close"].idxmin(), "timestamp"
                            ],
                        ],
                        "price": [
                            intraday_display["close"].max(),
                            intraday_display["close"].min(),
                        ],
                        "label": [
                            f"最高 ${intraday_display['close'].max():.2f}",
                            f"最低 ${intraday_display['close'].min():.2f}",
                        ],
                    }
                )
                intraday_figure.add_trace(
                    go.Scatter(
                        x=turning_points["timestamp"],
                        y=turning_points["price"],
                        mode="markers+text",
                        marker=dict(
                            color=["#ef4444", "#16a34a"],
                            size=9,
                            line=dict(color="#ffffff", width=1.5),
                        ),
                        text=turning_points["label"],
                        textposition=["top center", "bottom center"],
                        textfont=dict(size=12),
                        showlegend=False,
                        hovertemplate=(
                            "%{x|%H:%M}<br>%{text}<extra></extra>"
                        ),
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
                    y=display_last_close,
                    line_dash="dash",
                    line_color="#ef4444",
                    opacity=0.65,
                    annotation_text=f"前收 ${display_last_close:.2f}",
                    annotation_position="bottom right",
                )
                visible_min = min(
                    intraday_display["close"].min(),
                    intraday_display["average"].min(),
                    display_actual_intraday["close"].min()
                    if not display_actual_intraday.empty
                    else float("inf"),
                    display_last_close,
                )
                visible_max = max(
                    intraday_display["close"].max(),
                    intraday_display["average"].max(),
                    display_actual_intraday["close"].max()
                    if not display_actual_intraday.empty
                    else float("-inf"),
                    display_last_close,
                )
                price_span = max(visible_max - visible_min, display_last_close * 0.002)
                y_min = visible_min - price_span * 0.12
                y_max = visible_max + price_span * 0.12
                change_min = y_min / display_last_close - 1
                change_max = y_max / display_last_close - 1
                percent_ticks = [
                    change_min + (change_max - change_min) * index / 4
                    for index in range(5)
                ]
                intraday_figure.update_layout(
                    title=dict(
                        text=(
                            f"{etf_display_name(selected_symbol)} · "
                            f"{display_forecast_date.date()} "
                            + (
                                "5分钟模型预测"
                                if minute_model_available
                                else "日线OHLC情景路径"
                            )
                        ),
                        x=0,
                        xanchor="left",
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
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="left",
                        x=0,
                    ),
                    height=460,
                    margin=dict(l=10, r=65, t=95, b=10),
                )
                st.plotly_chart(intraday_figure, width="stretch")
                if minute_model_available:
                    st.caption(
                        "红点和绿点标识 5 分钟序列模型输出路径中的最高与最低预测值；"
                        "不再使用随机 OHLC 情景尖点。"
                    )
                    if not display_actual_intraday.empty:
                        st.caption(
                            f"黑线为 {selected_symbol} 截至所选时点的真实 5 分钟收盘价，"
                            "仅用于事后对照；不参与该日盘前预测训练。"
                        )
                else:
                    st.caption(
                        "红点和绿点分别标识预测最高价与最低价；其分钟位置由情景路径确定，"
                        "用于展示次日 OHLC 区间，不代表模型能精确预测极值发生时刻。"
                    )

                st.markdown("#### 共同样本外回测比较")
                if minute_model_available:
                    comparison = compare_backtests_on_common_dates(
                        forecast.backtest, minute_backtest
                    )
                    if comparison.empty:
                        st.warning("两套模型没有共同的样本外交易日，暂时无法直接比较。")
                    else:
                        daily_row = comparison.loc[
                            comparison["model"] == "日线OHLC基准"
                        ].iloc[0]
                        minute_row = comparison.loc[
                            comparison["model"] == "5分钟序列模型"
                        ].iloc[0]
                        direction_delta = (
                            minute_row["direction_accuracy"]
                            - daily_row["direction_accuracy"]
                        )
                        close_mae_delta = (
                            minute_row["close_mae"] - daily_row["close_mae"]
                        )
                        comparison_columns = st.columns(4)
                        comparison_columns[0].metric(
                            "分钟方向准确率",
                            f"{minute_row['direction_accuracy']:.1%}",
                            f"较日线 {direction_delta:+.1%}",
                        )
                        comparison_columns[1].metric(
                            "分钟收盘 MAE",
                            f"{minute_row['close_mae']:.2%}",
                            f"较日线 {close_mae_delta:+.2%}",
                            delta_color="inverse",
                        )
                        comparison_columns[2].metric(
                            "分钟高/低 MAE",
                            f"{minute_row['high_mae']:.2%} / {minute_row['low_mae']:.2%}",
                        )
                        comparison_columns[3].metric(
                            "共同回测日", f"{int(minute_row['sessions']):,}"
                        )

                        comparison_display = comparison.copy()
                        comparison_display["模型"] = comparison_display.pop("model")
                        for source, target in (
                            ("direction_accuracy", "方向准确率"),
                            ("open_mae", "开盘MAE"),
                            ("high_mae", "最高MAE"),
                            ("low_mae", "最低MAE"),
                            ("close_mae", "收盘MAE"),
                        ):
                            comparison_display[target] = comparison_display.pop(source).map(
                                lambda value: f"{value:.2%}"
                            )
                        comparison_display["回测日"] = comparison_display.pop("sessions")
                        comparison_display["共同区间"] = (
                            comparison_display.pop("start").dt.strftime("%Y-%m-%d")
                            + " — "
                            + comparison_display.pop("end").dt.strftime("%Y-%m-%d")
                        )
                        st.dataframe(
                            comparison_display,
                            hide_index=True,
                            width="stretch",
                        )

                        daily_reference = forecast.backtest[
                            ["forecast_date", "predicted_close"]
                        ].rename(
                            columns={
                                "forecast_date": "date",
                                "predicted_close": "daily_predicted_close",
                            }
                        )
                        minute_reference = minute_backtest[
                            ["session_date", "actual_close", "predicted_close"]
                        ].rename(
                            columns={
                                "session_date": "date",
                                "predicted_close": "minute_predicted_close",
                            }
                        )
                        aligned_backtest = minute_reference.merge(
                            daily_reference, on="date", how="inner"
                        ).tail(160)
                        backtest_figure = go.Figure()
                        for column, name, color, dash in (
                            ("actual_close", "实际收盘", "#111827", "solid"),
                            (
                                "minute_predicted_close",
                                "5分钟模型",
                                "#7c3aed",
                                "dot",
                            ),
                            (
                                "daily_predicted_close",
                                "日线OHLC基准",
                                "#1683ff",
                                "dash",
                            ),
                        ):
                            backtest_figure.add_trace(
                                go.Scatter(
                                    x=aligned_backtest["date"],
                                    y=aligned_backtest[column],
                                    mode="lines",
                                    line=dict(color=color, width=1.8, dash=dash),
                                    name=name,
                                )
                            )
                        backtest_figure.update_layout(
                            title="共同交易日预测收盘比较（最近160日）",
                            xaxis_title="预测交易日",
                            yaxis_title="收盘价",
                            hovermode="x unified",
                            legend=dict(orientation="h", y=1.12, x=0),
                            height=350,
                            margin=dict(l=10, r=10, t=55, b=10),
                        )
                        st.plotly_chart(backtest_figure, width="stretch")
                        st.caption(
                            "两套模型仅在共同样本外交易日比较；MAE 越低越好，"
                            "方向准确率越高越好。"
                        )
                else:
                    st.caption(
                        "取历史末段 20% 样本并按时间拆成 5 折；每折只使用该折开始日期"
                        "之前的数据训练，再预测后续交易日。"
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

                st.markdown(
                    "#### 日线基准可解释化"
                    if minute_model_available
                    else "#### 可解释化预测"
                )
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
                if minute_model_available:
                    interval_left.metric(
                        "分钟预测价格区间",
                        f"${display_low:,.2f} – ${display_high:,.2f}",
                    )
                    interval_right.metric(
                        "分钟回测覆盖区间",
                        f"{minute_metrics['backtest_start']} – "
                        f"{minute_metrics['backtest_end']}",
                    )
                    st.info(
                        "主分时图和顶部 OHLC 均来自 5 分钟序列模型；"
                        "日线 OHLC 模型仅作为准确率基准及解释参照。"
                    )
                    download_frame = minute_forecast
                    download_suffix = "minute_forecast"
                else:
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
                        "不是分钟级模型预测；预测日期按工作日估算。"
                    )
                    download_frame = forecast.intraday
                    download_suffix = "scenario_forecast"
                st.download_button(
                    "下载预测分时数据 CSV",
                    data=download_frame.to_csv(index=False).encode("utf-8"),
                    file_name=(
                        f"{selected_symbol}_{display_forecast_date.date()}_"
                        f"{download_suffix}.csv"
                    ),
                    mime="text/csv",
                )
            except Exception as error:
                st.error(f"行情预测生成失败：{error}")
        else:
            if minute_model_available:
                st.info(
                    "点击“生成预测”后，将加载 5 分钟模型主预测，并训练日线 OHLC 基准"
                    "用于共同样本外比较。"
                )
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
                "horizon_days",
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
            "预测周期（交易日）",
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
        importance_horizon = st.selectbox(
            "预测周期（交易日）",
            sorted(importance["horizon_days"].unique())
            if "horizon_days" in importance
            else [config.model.horizon_days],
            key="importance_horizon",
        )
        selected_importance = (
            importance.loc[
                (importance["symbol"] == importance_symbol)
                & (
                    importance["horizon_days"] == importance_horizon
                    if "horizon_days" in importance
                    else True
                )
            ]
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
            by_kind = by_kind.loc[
                (by_kind["symbol"] == importance_symbol)
                & (
                    by_kind["horizon_days"] == importance_horizon
                    if "horizon_days" in by_kind
                    else True
                )
            ].copy()
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
