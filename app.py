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
from etf_mvp.pipeline import run_pipeline, train_models

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


def format_feature_name(name: str) -> str:
    return name.replace("__", " · ").replace("_", " ")


def period_start(max_date: pd.Timestamp, period: str) -> pd.Timestamp | None:
    offsets = {
        "6个月": pd.DateOffset(months=6),
        "1年": pd.DateOffset(years=1),
        "3年": pd.DateOffset(years=3),
        "5年": pd.DateOffset(years=5),
    }
    return max_date - offsets[period] if period in offsets else None


config = load_config()

st.sidebar.title("研究控制台")
st.sidebar.caption("调整参数后可重新训练模型")
selected_targets = st.sidebar.multiselect(
    "预测标的",
    options=list(config.universe.symbols),
    default=list(config.universe.targets),
    max_selections=4,
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
    ["使用现有行情", "重新生成演示数据", "Tiingo 真实日线"],
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
                if not os.getenv("TIINGO_API_KEY"):
                    raise ValueError("请先在终端设置 TIINGO_API_KEY")
                run_pipeline(runtime_config, provider_name="tiingo")
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
            label=f"{row.symbol} · 未来 {row.horizon_days} 日",
            value=f"{row.probability_up:.1%}",
            delta=direction_text,
            delta_color="normal" if row.signal == "UP" else "inverse",
        )

    chart_left, chart_right = st.columns([1.35, 1])
    with chart_left:
        st.subheader("上涨概率对比")
        probability_figure = px.bar(
            display_predictions,
            x="symbol",
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
            f"**相对最强：** {strongest['symbol']}，"
            f"上涨概率 {strongest['probability_up']:.1%}"
        )
        st.write(
            f"**相对最弱：** {weakest['symbol']}，"
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
        selected_symbol = control_left.selectbox("查看 ETF", available_symbols)
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
                    name=selected_symbol,
                )
            ]
        )
        candle_figure.update_layout(
            title=f"{selected_symbol} 日线走势",
            xaxis_rangeslider_visible=False,
            yaxis_title="价格",
            height=470,
            margin=dict(l=10, r=10, t=45, b=10),
        )
        st.plotly_chart(candle_figure, width="stretch")

        compare_symbols = st.multiselect(
            "归一化走势对比",
            available_symbols,
            default=available_symbols[:3],
            max_selections=5,
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
        2. 选择现有行情、演示数据或 Tiingo，然后点击 **运行研究模型**。
        3. 在各标签页查看概率、历史走势、样本外指标和特征重要性。

        **指标提示**
        - `ROC AUC` 越接近 1，模型排序能力越强；约 0.5 表示接近随机。
        - `Brier` 越低，概率预测的校准误差越小。
        - 单次概率不能代表确定收益，应结合长期样本外表现判断。
        """
    )
    st.warning("本项目仅用于模型研究和学习，不构成投资建议，也不会发送任何交易指令。")
