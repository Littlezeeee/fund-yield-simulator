# main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="理财收益率模拟器", layout="wide")
plt_cfg = dict(font=dict(family="SimHei"))

# 中文

st.markdown("<style>html, body, [class*='css']  { font-family: SimHei, sans-serif; }</style>", unsafe_allow_html=True)

@st.cache_data
def load_bond_data():
    return pd.read_pickle('./data/885005.plk')

def generate_fixed_nav(name, annual_rate, index):
    df = pd.DataFrame(index=index, columns=[name])
    daily_rate = annual_rate / 100 / 252
    value = 1.0
    for date in index:
        df.loc[date, name] = value
        value *= (1 + daily_rate)
    return df

def generate_combined_nav(components, weights):
    nav_df = pd.concat(components, axis=1).dropna()
    returns = nav_df.pct_change().fillna(0)
    combined = (returns @ weights).add(1).cumprod()
    return combined

def calculate_annualized_distribution(nav_series, hold_days):
    returns = nav_series.pct_change().dropna()
    hold_returns = (1 + returns).rolling(window=hold_days).apply(np.prod, raw=True) - 1
    ann_returns = (1 + hold_returns) ** (252 / hold_days) - 1
    return ann_returns.dropna()

def calc_max_drawdown_info(nav_series):
    """
    计算净值序列最大回撤及其区间
    返回：最大回撤（bp）、开始日期、结束日期、持续天数
    """
    cumulative_max = nav_series.cummax()
    drawdown = (nav_series - cumulative_max) / cumulative_max
    min_dd = drawdown.min()

    # 找到最大回撤结束点（最低点）
    end_date = drawdown.idxmin()
    # 在这之前找最高点
    start_date = nav_series.loc[:end_date].idxmax()
    duration = (end_date - start_date).days

    return round(min_dd * 10000, 2), start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), duration



# 加载数据
bond_df = load_bond_data()
dates = bond_df.index

# ==== 控制面板 ====
st.sidebar.header("📌 固定资产收益率设置")
deposit_rate = st.sidebar.slider("存款年化收益率（%）", 0.00, 5.00, 1.75, step=0.01, format="%.2f")
plan_rate = st.sidebar.slider("资管计划年化收益率（%）", 0.00, 6.00, 3.00, step=0.01, format="%.2f")

st.sidebar.header("📌 资产配置权重设置（%）")

bond_weight = st.sidebar.number_input("债券基金", min_value=0, max_value=100, value=40, step=1)
deposit_weight = st.sidebar.number_input("存款", min_value=0, max_value=100, value=30, step=1)
plan_weight = st.sidebar.number_input("资管计划", min_value=0, max_value=100, value=30, step=1)

total_weight = bond_weight + deposit_weight + plan_weight
st.sidebar.markdown(f"🔍 当前总权重：**{total_weight}%**")

if total_weight != 100:
    st.sidebar.error("⚠️ 权重之和必须为 100%，请调整")
    st.stop()


hold_days = st.sidebar.number_input("持有期（交易日）", min_value=1, max_value=252, value=30, step=1)


# ==== 生成净值 ====
df_deposit = generate_fixed_nav("存款", deposit_rate, dates)
df_plan = generate_fixed_nav("资管计划", plan_rate, dates)
components = [bond_df, df_deposit, df_plan]
weights = np.array([bond_weight, deposit_weight, plan_weight]) / 100

portfolio_nav = generate_combined_nav(components, weights)
annual_ret = calculate_annualized_distribution(portfolio_nav, hold_days)

# 转换为百分比形式
ret_percent = annual_ret * 100
ret_percent_sorted = np.sort(ret_percent)
cdf = np.arange(1, len(ret_percent_sorted) + 1) / len(ret_percent_sorted)

# ==== 画图 ====
st.title("📊 理财产品收益率分布模拟器")

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=ret_percent,
    nbinsx=60,
    marker_color='skyblue',
    name="年化收益率分布",
    hovertemplate='年化收益率: %{x:.2f}%<br>频数: %{y}<extra></extra>',
))

fig.update_layout(
    title=f"组合持有期（{hold_days}天）年化收益率分布",
    xaxis_title="年化收益率 (%)",
    yaxis_title="频数",
    template="simple_white",
    font=dict(family="SimHei"),
    bargap=0.05
)

st.plotly_chart(fig, use_container_width=True)

# ==== 累计概率图 ====
st.subheader("📈 累计分布 (点击横坐标 x 可估算小于 x 的概率)")
fig_cdf = go.Figure()
fig_cdf.add_trace(go.Scatter(
    x=ret_percent_sorted,
    y=cdf,
    mode='lines',
    name='累计概率',
    hovertemplate='收益率: %{x:.2f}%<br>概率: %{y:.2%}<extra></extra>'
))
fig_cdf.update_layout(
    xaxis_title="年化收益率 (%)",
    yaxis_title="累计概率",
    template="simple_white",
    font=dict(family="SimHei")
)
st.plotly_chart(fig_cdf, use_container_width=True)

# ==== 净值曲线图 ====
st.subheader("📉 投资组合净值曲线")
fig_nav = go.Figure()
fig_nav.add_trace(go.Scatter(
    x=portfolio_nav.index,
    y=portfolio_nav.values,
    mode='lines',
    name='组合净值',
    line=dict(color='green'),
    hovertemplate='日期: %{x}<br>净值: %{y:.4f}<extra></extra>'
))
fig_nav.update_layout(
    xaxis_title="日期",
    yaxis_title="组合净值",
    template="simple_white",
    font=dict(family="SimHei"),
    height=400
)
st.plotly_chart(fig_nav, use_container_width=True)


# ==== 描述统计 ====
st.subheader("🔍 年化收益率分布统计（%）")

desc = ret_percent.describe(percentiles=[.05, .25, .5, .75, .95]).round(2)
mean = desc['mean']
daily_returns = portfolio_nav.pct_change().dropna()
annual_std = daily_returns.std() * np.sqrt(252)
sharpe = round((mean / 100) / annual_std if annual_std != 0 else np.nan, 2)
max_drawdown, dd_start, dd_end, dd_duration = calc_max_drawdown_info(portfolio_nav)
return_drawdown_ratio = -round(mean / (max_drawdown / 100) if max_drawdown != 0 else np.nan, 2)

extra_metrics = pd.DataFrame({
    "统计": [
        "Sharpe 比例", 
        "最大回撤（bp）", 
        "收益回撤比", 
        "回撤开始日期", 
        "回撤结束日期", 
        "回撤持续天数"
    ],
    "值": [
        sharpe,
        max_drawdown,
        return_drawdown_ratio,
        dd_start,
        dd_end,
        dd_duration
    ]
})


desc_df = pd.DataFrame(desc)
desc_df.columns = ["统计"]
st.dataframe(desc_df)
st.markdown("### 📎 附加指标")
st.table(extra_metrics.set_index("统计"))


