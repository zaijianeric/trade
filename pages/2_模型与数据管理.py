from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from etf_mvp.inventory import (
    scan_data_sources,
    scan_dependencies,
    scan_ensemble,
    scan_feature_ablation,
    scan_intraday_quality,
    scan_intraday_quality_comparison,
    scan_models,
    scan_news_quality,
    scan_rolling_tuning,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(
    page_title="模型与数据管理 · ETF Research Lab",
    page_icon="🧩",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    [data-testid="stSidebar"] { background: #101827; }
    [data-testid="stSidebar"] * { color: #f8fafc; }
    [data-testid="stMainBlockContainer"] { max-width: 1500px; padding-top: 2.4rem; }
    .manage-hero {
        padding: 25px 28px;
        border-radius: 18px;
        color: white;
        background: linear-gradient(120deg, #111827, #173b78);
        margin-bottom: 22px;
    }
    .manage-hero h1 { margin: 0 0 7px; font-size: 2rem; }
    .manage-hero p { margin: 0; color: #cbd5e1; }
    .status-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 18px 20px;
        box-shadow: 0 7px 20px rgba(15, 23, 42, .04);
    }
    .status-label { color: #64748b; font-size: .82rem; }
    .status-value { color: #0f172a; font-size: 1.75rem; font-weight: 750; margin-top: 5px; }
    .section-note { color: #64748b; margin: -8px 0 14px; }
    div[data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 13px;
        overflow: hidden;
    }
    </style>
    <div class="manage-hero">
      <h1>模型与数据管理</h1>
      <p>集中查看当前模型、核心依赖和数据源状态；本期仅提供只读展示。</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, ttl=60)
def load_inventory(
    root: str,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    project_root = Path(root)
    return (
        pd.DataFrame(scan_models(project_root)),
        pd.DataFrame(scan_data_sources(project_root)),
        pd.DataFrame(scan_dependencies(project_root)),
        pd.DataFrame(scan_intraday_quality(project_root)),
        pd.DataFrame(scan_intraday_quality_comparison(project_root)),
        pd.DataFrame(scan_feature_ablation(project_root)),
        pd.DataFrame(scan_rolling_tuning(project_root)),
        pd.DataFrame(scan_ensemble(project_root)),
        pd.DataFrame(scan_news_quality(project_root)),
    )


(
    models,
    sources,
    dependencies,
    quality,
    quality_comparison,
    feature_ablation,
    rolling_tuning,
    ensemble,
    news_quality,
) = load_inventory(str(PROJECT_ROOT))
available_sources = int((sources["状态"] == "可用").sum()) if not sources.empty else 0
latest_update = sources.loc[sources["状态"] == "可用", "文件更新"].max() if available_sources else "—"

summary_columns = st.columns(4)
summary_values = (
    ("模型产物", f"{len(models)} 个"),
    ("可用数据集", f"{available_sources} / {len(sources)}"),
    ("核心依赖", f"{len(dependencies)} 项"),
    ("最近文件更新", str(latest_update)),
)
for column, (label, value) in zip(summary_columns, summary_values, strict=True):
    column.markdown(
        f'<div class="status-card"><div class="status-label">{label}</div>'
        f'<div class="status-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.subheader("当前模型", divider="gray")
st.markdown('<p class="section-note">版本为模型算法包版本，大小与更新时间来自当前 Joblib 产物。</p>', unsafe_allow_html=True)
if models.empty:
    st.info("尚未发现模型产物。")
else:
    st.dataframe(
        models,
        hide_index=True,
        width="stretch",
        column_config={
            "文件": st.column_config.TextColumn(width="large"),
            "用途": st.column_config.TextColumn(width="medium"),
        },
    )

st.subheader("数据源", divider="gray")
st.markdown('<p class="section-note">“最新数据”是文件内最新观测日期；“文件更新”是本地文件最后修改时间。</p>', unsafe_allow_html=True)
for category in ("标的历史行情", "经济数据", "新闻", "基本面"):
    category_frame = sources.loc[sources["分类"] == category].drop(columns="分类")
    if category_frame.empty:
        continue
    st.markdown(f"#### {category}")
    st.dataframe(
        category_frame,
        hide_index=True,
        width="stretch",
        column_config={
            "记录数": st.column_config.NumberColumn(format="localized"),
            "状态": st.column_config.TextColumn(width="small"),
            "提供方": st.column_config.TextColumn(width="large"),
        },
    )

if not news_quality.empty:
    st.markdown("#### 新闻数据质量")
    st.dataframe(
        news_quality,
        hide_index=True,
        width="stretch",
        column_config={
            "记录数": st.column_config.NumberColumn(format="localized"),
            "活跃日期": st.column_config.NumberColumn(format="localized"),
            "距行情截止": st.column_config.NumberColumn(
                format="%d 天", help="新闻最新可用日与行情截止日的日历天差"
            ),
        },
    )

st.subheader("分钟数据质量", divider="gray")
st.markdown('<p class="section-note">质量评分按交易所日历中的实际观测K线/应有K线计算；短缺口最多填补3根，长缺口交易日不进入训练。</p>', unsafe_allow_html=True)
if quality.empty:
    st.info("尚未生成分钟数据质量报告，可运行 `uv run etf-mvp intraday-quality`。")
else:
    st.dataframe(
        quality,
        hide_index=True,
        width="stretch",
        column_config={
            "质量评分": st.column_config.ProgressColumn(
                format="%.1f", min_value=0, max_value=100
            ),
            "K线完整率": st.column_config.ProgressColumn(
                format="percent", min_value=0, max_value=1
            ),
            "可训练交易日": st.column_config.NumberColumn(format="localized"),
            "清洗后记录": st.column_config.NumberColumn(format="localized"),
        },
    )
    if not quality_comparison.empty:
        st.markdown("#### 清洗前后同区间回测")
        st.dataframe(
            quality_comparison,
            hide_index=True,
            width="stretch",
            column_config={
                "清洗前准确率": st.column_config.NumberColumn(format="percent"),
                "清洗后准确率": st.column_config.NumberColumn(format="percent"),
                "变化": st.column_config.NumberColumn(format="percent"),
            },
        )

st.subheader("特征增强消融", divider="gray")
st.markdown('<p class="section-note">基线与增强模型使用相同数据、时间区间和随机种子；准确率提升1.5个百分点或误差显著下降视为达标，未达标则自动保留基线模型。</p>', unsafe_allow_html=True)
if feature_ablation.empty:
    st.info("尚未生成特征消融结果，请重新训练日线与分钟模型。")
else:
    st.dataframe(
        feature_ablation,
        hide_index=True,
        width="stretch",
        column_config={
            "基线准确率": st.column_config.NumberColumn(format="percent"),
            "增强准确率": st.column_config.NumberColumn(format="percent"),
            "准确率变化": st.column_config.NumberColumn(format="percent"),
            "辅助指标变化": st.column_config.NumberColumn(format="%.4f"),
        },
    )

st.subheader("滚动调参", divider="gray")
st.markdown('<p class="section-note">仅使用外层测试区间之前的数据进行内层扩展窗口搜索；新参数通过同区间外层验收后才替换固定参数。</p>', unsafe_allow_html=True)
if rolling_tuning.empty:
    st.info("尚未生成滚动调参结果，请重新训练日线与分钟模型。")
else:
    st.dataframe(
        rolling_tuning,
        hide_index=True,
        width="stretch",
        column_config={
            "固定参数准确率": st.column_config.NumberColumn(format="percent"),
            "调参后准确率": st.column_config.NumberColumn(format="percent"),
            "准确率变化": st.column_config.NumberColumn(format="percent"),
            "辅助指标变化": st.column_config.NumberColumn(format="%.4f"),
        },
    )

st.subheader("多模型集成", divider="gray")
st.markdown('<p class="section-note">日线组合 LightGBM、ExtraTrees、逻辑回归并校准概率；分钟组合 LightGBM、ExtraTrees、HistGradientBoosting。权重仅由外层测试前的滚动窗口确定。</p>', unsafe_allow_html=True)
if ensemble.empty:
    st.info("尚未生成集成结果，请重新训练日线与分钟模型。")
else:
    st.dataframe(
        ensemble,
        hide_index=True,
        width="stretch",
        column_config={
            "单模型准确率": st.column_config.NumberColumn(format="percent"),
            "集成准确率": st.column_config.NumberColumn(format="percent"),
            "准确率变化": st.column_config.NumberColumn(format="percent"),
            "辅助指标变化": st.column_config.NumberColumn(format="%.4f"),
            "权重": st.column_config.TextColumn(width="large"),
        },
    )

with st.expander("核心依赖版本", expanded=False):
    st.caption("展示项目直接依赖及当前运行环境中的实际安装版本。")
    st.dataframe(dependencies, hide_index=True, width="stretch")

if st.button("刷新展示", type="secondary"):
    st.cache_data.clear()
    st.rerun()
