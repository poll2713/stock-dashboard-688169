# -*- coding: utf-8 -*-
"""
石头科技 (688169.SH) 股价数据分析与技术指标计算
===============================================
任务1: 数据基础诊断分析（缺失值检查 + 描述性统计）
任务3: Python计算 RSI / MACD / 布林带 并可视化
任务4: 扩展指标 OBV（能量潮指标）介绍与计算
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os, math

# ============================================================
# 全局设置
# ============================================================
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# A股颜色惯例：红涨绿跌
UP_COLOR = '#e8554e'
DOWN_COLOR = '#1ba653'
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(DATA_DIR, '688169_石头科技_daily_20250704_20260704.csv')
OUTPUT_DIR = os.path.join(DATA_DIR, 'charts')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 任务1：加载数据 & 基础诊断分析
# ============================================================
print("=" * 70)
print("任务1：数据基础诊断分析")
print("=" * 70)

# 1) 加载已存储的股价数据
df = pd.read_csv(CSV_FILE, encoding='utf-8-sig')
# 日期转换为标准格式
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
# 按日期升序排列
df = df.sort_values('trade_date').reset_index(drop=True)
df.set_index('trade_date', inplace=True)

print(f"\n【数据概况】")
print(f"  股票代码: {df['ts_code'].iloc[0]}")
print(f"  数据区间: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}")
print(f"  交易日数: {len(df)} 天")
print(f"  字段数量: {len(df.columns)} 个")
print(f"  字段列表: {', '.join(df.columns)}")

# 2) 缺失值检查
print(f"\n【缺失值检查】")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
if missing.sum() == 0:
    print(f"  ✓ 所有字段均无缺失值，数据完整")
else:
    for col in df.columns:
        if missing[col] > 0:
            print(f"  ✗ {col}: 缺失 {missing[col]} 个 ({missing_pct[col]}%)")

# 3) 描述性统计量
print(f"\n【描述性统计量】")
desc = df[['open', 'high', 'low', 'close', 'vol', 'amount']].describe().round(2)
print(desc.to_string())

# 4) 补充统计
print(f"\n【补充统计】")
print(f"  收盘价均值:     {df['close'].mean():.2f} 元")
print(f"  收盘价中位数:   {df['close'].median():.2f} 元")
print(f"  收盘价标准差:   {df['close'].std():.2f} 元")
print(f"  收盘价偏度:     {df['close'].skew():.4f}")
print(f"  收盘价峰度:     {df['close'].kurt():.4f}")
print(f"  区间最高价:     {df['high'].max():.2f} 元 ({df['high'].idxmax().strftime('%Y-%m-%d')})")
print(f"  区间最低价:     {df['low'].min():.2f} 元 ({df['low'].idxmin().strftime('%Y-%m-%d')})")
print(f"  日均成交量:     {df['vol'].mean():,.0f} 手")
print(f"  日均成交额:     {df['amount'].mean()/100:,.0f} 千元")

# 涨跌统计
up_days = (df['pct_chg'] > 0).sum()
down_days = (df['pct_chg'] < 0).sum()
flat_days = (df['pct_chg'] == 0).sum()
print(f"  上涨天数:       {up_days} 天 ({up_days/len(df)*100:.1f}%)")
print(f"  下跌天数:       {down_days} 天 ({down_days/len(df)*100:.1f}%)")
print(f"  平盘天数:       {flat_days} 天")

# 日收益率统计
daily_returns = df['pct_chg']
print(f"\n【日涨跌幅统计】")
print(f"  平均日涨跌幅:   {daily_returns.mean():.4f}%")
print(f"  日涨跌幅标准差: {daily_returns.std():.4f}%")
print(f"  最大单日涨幅:   +{daily_returns.max():.2f}% ({daily_returns.idxmax().strftime('%Y-%m-%d')})")
print(f"  最大单日跌幅:   {daily_returns.min():.2f}% ({daily_returns.idxmin().strftime('%Y-%m-%d')})")

# 年化波动率（假设252个交易日）
annual_vol = daily_returns.std() * math.sqrt(252)
print(f"  年化波动率:     {annual_vol:.2f}%")

# 最大回撤
cummax = df['close'].cummax()
drawdown = (df['close'] - cummax) / cummax
max_dd = drawdown.min()
print(f"  最大回撤:       {max_dd*100:.2f}%")

# ============================================================
# 任务3：技术指标计算
# ============================================================
print("\n" + "=" * 70)
print("任务3：技术指标计算 (RSI / MACD / 布林带)")
print("=" * 70)

close = df['close']
high = df['high']
low = df['low']
volume = df['vol']

# ---- 3.1 RSI (相对强弱指数) ----
# 计算方法：RSI = 100 - 100/(1+RS)
#   RS = N日内平均涨幅 / N日内平均跌幅（Wilder平滑法）
print("\n--- RSI (相对强弱指数, 参数=14) ---")

delta = close.diff()
gain = delta.where(delta > 0, 0.0)
loss = (-delta).where(delta < 0, 0.0)

# Wilder平滑法
avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
rs = avg_gain / avg_loss
df['RSI_14'] = (100 - (100 / (1 + rs))).round(2)

print(f"  最新RSI值:      {df['RSI_14'].iloc[-1]:.2f}")
print(f"  RSI最大值:      {df['RSI_14'].max():.2f} ({df['RSI_14'].idxmax().strftime('%Y-%m-%d')})")
print(f"  RSI最小值:      {df['RSI_14'].min():.2f} ({df['RSI_14'].idxmin().strftime('%Y-%m-%d')})")
overbought = (df['RSI_14'] > 70).sum()
oversold = (df['RSI_14'] < 30).sum()
print(f"  超买天数(RSI>70): {overbought} 天")
print(f"  超卖天数(RSI<30): {oversold} 天")

# ---- 3.2 MACD (指数平滑异同移动平均线) ----
# 计算方法：
#   DIF = EMA(12) - EMA(26)
#   DEA = EMA(DIF, 9)
#   MACD柱 = (DIF - DEA) × 2
print("\n--- MACD (参数 12, 26, 9) ---")

ema12 = close.ewm(span=12, adjust=False).mean()
ema26 = close.ewm(span=26, adjust=False).mean()
df['DIF'] = (ema12 - ema26).round(4)
df['DEA'] = df['DIF'].ewm(span=9, adjust=False).mean().round(4)
df['MACD'] = ((df['DIF'] - df['DEA']) * 2).round(4)

# 统计金叉死叉
dif_above = df['DIF'] > df['DEA']
golden_cross = ((~dif_above.shift(1, fill_value=False)) & dif_above).sum()
death_cross = (dif_above.shift(1, fill_value=False) & (~dif_above)).sum()
print(f"  最新DIF:        {df['DIF'].iloc[-1]:.4f}")
print(f"  最新DEA:        {df['DEA'].iloc[-1]:.4f}")
print(f"  最新MACD柱:     {df['MACD'].iloc[-1]:.4f}")
print(f"  金叉次数(DIF上穿DEA): {golden_cross} 次")
print(f"  死叉次数(DIF下穿DEA): {death_cross} 次")

# ---- 3.3 布林带 (Bollinger Bands) ----
# 计算方法：
#   中轨 = MA(20)
#   上轨 = MA(20) + 2 × σ
#   下轨 = MA(20) - 2 × σ
print("\n--- BOLL (布林带, 参数 20, 2) ---")

df['BOLL_MID'] = close.rolling(window=20).mean().round(2)
df['BOLL_STD'] = close.rolling(window=20).std().round(4)
df['BOLL_UP'] = (df['BOLL_MID'] + 2 * df['BOLL_STD']).round(2)
df['BOLL_LOW'] = (df['BOLL_MID'] - 2 * df['BOLL_STD']).round(2)
df['BOLL_WIDTH'] = ((df['BOLL_UP'] - df['BOLL_LOW']) / df['BOLL_MID'] * 100).round(2)

print(f"  最新中轨:       {df['BOLL_MID'].iloc[-1]:.2f}")
print(f"  最新上轨:       {df['BOLL_UP'].iloc[-1]:.2f}")
print(f"  最新下轨:       {df['BOLL_LOW'].iloc[-1]:.2f}")
print(f"  最新带宽:       {df['BOLL_WIDTH'].iloc[-1]:.2f}%")
print(f"  平均带宽:       {df['BOLL_WIDTH'].mean():.2f}%")

# 触及上轨次数（收盘价 >= 上轨）
touch_up = (close >= df['BOLL_UP']).sum()
touch_low = (close <= df['BOLL_LOW']).sum()
print(f"  收盘触及上轨天数: {touch_up} 天")
print(f"  收盘触及下轨天数: {touch_low} 天")

# ============================================================
# 任务4：扩展指标 - OBV (能量潮指标)
# ============================================================
print("\n" + "=" * 70)
print("任务4：扩展指标 OBV (On Balance Volume, 能量潮指标)")
print("=" * 70)

# OBV计算方法：
#   如果今日收盘价 > 昨日收盘价: OBV = 昨日OBV + 今日成交量
#   如果今日收盘价 < 昨日收盘价: OBV = 昨日OBV - 今日成交量
#   如果今日收盘价 = 昨日收盘价: OBV = 昨日OBV
obv = [0] * len(df)
for i in range(1, len(df)):
    if close.iloc[i] > close.iloc[i-1]:
        obv[i] = obv[i-1] + volume.iloc[i]
    elif close.iloc[i] < close.iloc[i-1]:
        obv[i] = obv[i-1] - volume.iloc[i]
    else:
        obv[i] = obv[i-1]
df['OBV'] = obv
df['OBV_MA20'] = df['OBV'].rolling(window=20).mean().round(0)

print(f"  最新OBV值:     {df['OBV'].iloc[-1]:,.0f}")
print(f"  OBV最高值:     {df['OBV'].max():,.0f} ({df['OBV'].idxmax().strftime('%Y-%m-%d')})")
print(f"  OBV最低值:     {df['OBV'].min():,.0f} ({df['OBV'].idxmin().strftime('%Y-%m-%d')})")
print(f"  OBV 20日均值:   {df['OBV_MA20'].iloc[-1]:,.0f}")

# ============================================================
# 可视化绘图
# ============================================================
print("\n" + "=" * 70)
print("开始生成可视化图形...")
print("=" * 70)

dates = df.index

# ---- 图1: 收盘价走势 + 描述性统计标注 ----
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates, close, color='#4a90d9', linewidth=1.2, label='收盘价')
ax.fill_between(dates, close, alpha=0.08, color='#4a90d9')
mean_line = close.mean()
ax.axhline(y=mean_line, color='#f5a623', linestyle='--', linewidth=1, label=f'均值 {mean_line:.2f}')
# 标注最高最低
ax.annotate(f"最高 {df['high'].max():.0f}", xy=(df['high'].idxmax(), df['high'].max()),
            fontsize=9, color=UP_COLOR, ha='center', xytext=(0, 12), textcoords='offset points')
ax.annotate(f"最低 {df['low'].min():.0f}", xy=(df['low'].idxmin(), df['low'].min()),
            fontsize=9, color=DOWN_COLOR, ha='center', xytext=(0, -15), textcoords='offset points')
ax.set_title('石头科技 (688169.SH) 收盘价走势', fontsize=14, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('价格 (元)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '01_price_trend.png'), dpi=150)
print("  ✓ 01_price_trend.png (收盘价走势)")
plt.close()

# ---- 图2: 日涨跌幅分布直方图 ----
fig, ax = plt.subplots(figsize=(12, 4))
bins = np.arange(math.floor(daily_returns.min()), math.ceil(daily_returns.max()) + 1, 1)
n_counts, bin_edges, patches = ax.hist(daily_returns, bins=bins, edgecolor='white', linewidth=0.5, alpha=0.85)
for patch, bin_left in zip(patches, bin_edges[:-1]):
    patch.set_facecolor(UP_COLOR if bin_left >= 0 else DOWN_COLOR)
ax.axvline(x=0, color='#333', linewidth=1, linestyle='-')
ax.axvline(x=daily_returns.mean(), color='#4a90d9', linewidth=1.5, linestyle='--', label=f'均值 {daily_returns.mean():.2f}%')
ax.set_title('日涨跌幅分布直方图', fontsize=13, fontweight='bold')
ax.set_xlabel('涨跌幅 (%)')
ax.set_ylabel('天数')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '02_return_dist.png'), dpi=150)
print("  ✓ 02_return_dist.png (日涨跌幅分布)")
plt.close()

# ---- 图3: RSI ----
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(dates, df['RSI_14'], color='#9b59b6', linewidth=1.2, label='RSI(14)')
ax.fill_between(dates, 70, 100, alpha=0.1, color=UP_COLOR, label='超买区 (>70)')
ax.fill_between(dates, 0, 30, alpha=0.1, color=DOWN_COLOR, label='超卖区 (<30)')
ax.axhline(y=70, color=UP_COLOR, linewidth=0.8, linestyle='--')
ax.axhline(y=30, color=DOWN_COLOR, linewidth=0.8, linestyle='--')
ax.axhline(y=50, color='#aaa', linewidth=0.5, linestyle=':')
ax.set_title('RSI 相对强弱指数 (14日)', fontsize=13, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('RSI')
ax.set_ylim(0, 100)
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '03_rsi.png'), dpi=150)
print("  ✓ 03_rsi.png (RSI指标)")
plt.close()

# ---- 图4: MACD ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), gridspec_kw={'height_ratios': [2, 1.2]}, sharex=True)
# 上: 收盘价
ax1.plot(dates, close, color='#4a90d9', linewidth=1.2, label='收盘价')
ax1.set_title('收盘价', fontsize=12, fontweight='bold')
ax1.set_ylabel('价格 (元)')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
# 下: MACD
bar_colors = [UP_COLOR if v >= 0 else DOWN_COLOR for v in df['MACD']]
ax2.bar(dates, df['MACD'], color=bar_colors, width=1, alpha=0.7, label='MACD柱')
ax2.plot(dates, df['DIF'], color='#e8554e', linewidth=1, label='DIF')
ax2.plot(dates, df['DEA'], color='#ffde00', linewidth=1, label='DEA')
ax2.axhline(y=0, color='#333', linewidth=0.8)
ax2.set_title('MACD 指标 (12,26,9)', fontsize=12, fontweight='bold')
ax2.set_xlabel('日期')
ax2.set_ylabel('MACD')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '04_macd.png'), dpi=150)
print("  ✓ 04_macd.png (MACD指标)")
plt.close()

# ---- 图5: 布林带 ----
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(dates, close, color='#4a90d9', linewidth=1.2, label='收盘价')
ax.plot(dates, df['BOLL_UP'], color=UP_COLOR, linewidth=1, linestyle='--', label='上轨')
ax.plot(dates, df['BOLL_MID'], color='#f5a623', linewidth=1, label='中轨(MA20)')
ax.plot(dates, df['BOLL_LOW'], color=DOWN_COLOR, linewidth=1, linestyle='--', label='下轨')
ax.fill_between(dates, df['BOLL_UP'], df['BOLL_LOW'], alpha=0.08, color='#4a90d9')
ax.set_title('布林带 Bollinger Bands (20, 2)', fontsize=13, fontweight='bold')
ax.set_xlabel('日期')
ax.set_ylabel('价格 (元)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '05_boll.png'), dpi=150)
print("  ✓ 05_boll.png (布林带)")
plt.close()

# ---- 图6: OBV 能量潮 ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={'height_ratios': [1.5, 1]}, sharex=True)
# 上: 收盘价
ax1.plot(dates, close, color='#4a90d9', linewidth=1.2, label='收盘价')
ax1.set_title('收盘价', fontsize=12, fontweight='bold')
ax1.set_ylabel('价格 (元)')
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)
# 下: OBV
obv_colors = []
for i in range(len(df)):
    if i > 0:
        obv_colors.append(UP_COLOR if df['OBV'].iloc[i] >= df['OBV'].iloc[i-1] else DOWN_COLOR)
    else:
        obv_colors.append(UP_COLOR)
ax2.bar(dates, df['OBV'], color=obv_colors, width=1, alpha=0.6)
ax2.plot(dates, df['OBV_MA20'], color='#4a90d9', linewidth=1.2, label='OBV MA20')
ax2.set_title('OBV 能量潮指标', fontsize=12, fontweight='bold')
ax2.set_xlabel('日期')
ax2.set_ylabel('OBV')
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.xaxis.set_major_locator(mdates.MonthLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, '06_obv.png'), dpi=150)
print("  ✓ 06_obv.png (OBV能量潮)")
plt.close()

# ---- 图7: 综合仪表盘 (4合1) ----
fig, axes = plt.subplots(4, 1, figsize=(16, 16), sharex=True,
                          gridspec_kw={'height_ratios': [3, 1, 1.5, 1.5]})
fig.suptitle('石头科技 (688169.SH) 技术指标综合分析', fontsize=16, fontweight='bold', y=0.98)

# 子图1: 收盘价 + 布林带
ax = axes[0]
ax.plot(dates, close, color='#4a90d9', linewidth=1.2, label='收盘价')
ax.plot(dates, df['BOLL_UP'], color=UP_COLOR, linewidth=0.8, linestyle='--', label='BOLL上轨')
ax.plot(dates, df['BOLL_MID'], color='#f5a623', linewidth=0.8, label='BOLL中轨')
ax.plot(dates, df['BOLL_LOW'], color=DOWN_COLOR, linewidth=0.8, linestyle='--', label='BOLL下轨')
ax.fill_between(dates, df['BOLL_UP'], df['BOLL_LOW'], alpha=0.06, color='#4a90d9')
ax.set_ylabel('价格 (元)')
ax.set_title('收盘价 + 布林带', fontsize=12)
ax.legend(loc='upper left', fontsize=8, ncol=4)
ax.grid(True, alpha=0.3)

# 子图2: 成交量
ax = axes[1]
vol_colors = [UP_COLOR if df['close'].iloc[i] >= df['open'].iloc[i] else DOWN_COLOR for i in range(len(df))]
ax.bar(dates, df['vol'], color=vol_colors, width=1, alpha=0.7)
ax.set_ylabel('成交量 (手)')
ax.set_title('成交量', fontsize=12)
ax.grid(True, alpha=0.3)

# 子图3: MACD
ax = axes[2]
bar_colors_macd = [UP_COLOR if v >= 0 else DOWN_COLOR for v in df['MACD']]
ax.bar(dates, df['MACD'], color=bar_colors_macd, width=1, alpha=0.7, label='MACD柱')
ax.plot(dates, df['DIF'], color='#e8554e', linewidth=1, label='DIF')
ax.plot(dates, df['DEA'], color='#ffde00', linewidth=1, label='DEA')
ax.axhline(y=0, color='#333', linewidth=0.8)
ax.set_ylabel('MACD')
ax.set_title('MACD (12,26,9)', fontsize=12)
ax.legend(loc='upper left', fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)

# 子图4: RSI
ax = axes[3]
ax.plot(dates, df['RSI_14'], color='#9b59b6', linewidth=1, label='RSI(14)')
ax.fill_between(dates, 70, 100, alpha=0.1, color=UP_COLOR)
ax.fill_between(dates, 0, 30, alpha=0.1, color=DOWN_COLOR)
ax.axhline(y=70, color=UP_COLOR, linewidth=0.6, linestyle='--')
ax.axhline(y=30, color=DOWN_COLOR, linewidth=0.6, linestyle='--')
ax.axhline(y=50, color='#aaa', linewidth=0.4, linestyle=':')
ax.set_ylabel('RSI')
ax.set_xlabel('日期')
ax.set_title('RSI 相对强弱指数', fontsize=12)
ax.set_ylim(0, 100)
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(OUTPUT_DIR, '07_dashboard.png'), dpi=150)
print("  ✓ 07_dashboard.png (综合仪表盘)")
plt.close()

# ============================================================
# 保存含指标的完整数据
# ============================================================
output_csv = os.path.join(DATA_DIR, '688169_指标数据_完整.csv')
df_output = df.copy()
df_output.index.name = 'trade_date'
df_output.to_csv(output_csv, encoding='utf-8-sig')
print(f"\n✓ 含所有指标的完整数据已保存: 688169_指标数据_完整.csv")

print("\n" + "=" * 70)
print("全部分析完成！")
print(f"  图表保存目录: {OUTPUT_DIR}")
print("=" * 70)
