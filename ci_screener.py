"""
独立筛选脚本（可用于 GitHub Actions）
- 仅需 2 次 Tushare API 调用
- 不需要本地缓存文件
- 输出 STOCK_LIST 格式
"""
import tushare as ts
import pandas as pd
import os, sys

def run(token):
    ts.set_token(token)
    pro = ts.pro_api()

    # 1. 交易日历
    cal = pro.trade_cal(start_date="20260601", end_date="20260729")
    trading_days = cal[cal['is_open'] == 1]['cal_date'].sort_values().tolist()
    END_DATE = trading_days[-1]

    # 2. 取前20个交易日（为了计算MA20）
    recent_dates = trading_days[-20:]
    START_DATE = recent_dates[0]

    print(f"📅 区间: {START_DATE} ~ {END_DATE}")

    # 3. 下载日线（1次调用=全市场）
    all_daily = []
    for dt in recent_dates:
        df = pro.daily(trade_date=dt)
        if df is not None and len(df) > 0:
            df['trade_date'] = str(dt)
            all_daily.append(df)
    daily = pd.concat(all_daily, ignore_index=True)
    print(f"📥 日线数据: {len(daily)} 条")

    # 4. 下载每日指标（1次调用=全市场）
    all_basic = []
    for dt in recent_dates:
        df = pro.daily_basic(trade_date=dt)
        if df is not None and len(df) > 0:
            df['trade_date'] = str(dt)
            all_basic.append(df)
    basic_data = pd.concat(all_basic, ignore_index=True)
    print(f"📥 指标数据: {len(basic_data)} 条")

    # 5. 股票基本信息（1次调用=永久缓存）
    stock_basic = pro.stock_basic(exchange='', list_status='L')

    # ====== 构建股票池 ======
    df_start = daily[daily['trade_date'] == START_DATE][['ts_code', 'close']].rename(columns={'close': 'close_start'})
    df_end = daily[daily['trade_date'] == END_DATE][['ts_code', 'close']].rename(columns={'close': 'close_end'})
    pool = pd.merge(df_start, df_end, on='ts_code', how='inner')
    pool['pct_change'] = round((pool['close_end'] - pool['close_start']) / pool['close_start'] * 100, 2)
    pool = pd.merge(pool, stock_basic[['ts_code', 'name', 'industry']], on='ts_code', how='left')

    # 当日涨跌幅
    latest_pct = daily[daily['trade_date'] == END_DATE][['ts_code', 'pct_chg']].rename(columns={'pct_chg': 'daily_pct'})
    pool = pd.merge(pool, latest_pct, on='ts_code', how='left')

    # 最新估值
    lb = basic_data[basic_data['trade_date'] == END_DATE][['ts_code', 'pe', 'pb', 'total_mv', 'turnover_rate']]
    pool = pd.merge(pool, lb, on='ts_code', how='left')
    pool['total_mv_yi'] = pool['total_mv'] / 10000

    # 最新成交额
    ld = daily[daily['trade_date'] == END_DATE][['ts_code', 'amount', 'vol']]
    pool = pd.merge(pool, ld, on='ts_code', how='left')
    pool['amount_yi'] = pool['amount'] / 100000

    # 均价
    pool['avg_price'] = pool['amount'] * 10 / pool['vol']

    # ====== 计算 MA5/MA10/MA20 ======
    def calc_ma(group, n):
        top = group.sort_values('trade_date', ascending=False).head(n)
        return top['close'].mean()

    ma_data = daily.groupby('ts_code').apply(
        lambda g: pd.Series({
            'ma5': calc_ma(g, 5), 'ma10': calc_ma(g, 10), 'ma20': calc_ma(g, 20)
        })
    ).reset_index()
    pool = pd.merge(pool, ma_data, on='ts_code', how='left')

    # 5日均额、20日均额
    amt_5d = daily.groupby('ts_code').apply(lambda g: g.sort_values('trade_date', ascending=False).head(5)['amount'].mean()).reset_index(name='amt_5d')
    amt_20d = daily.groupby('ts_code')['amount'].mean().reset_index().rename(columns={'amount': 'amt_20d'})
    pool = pd.merge(pool, amt_5d, on='ts_code', how='left')
    pool = pd.merge(pool, amt_20d, on='ts_code', how='left')

    # ====== 筛选条件 ======
    before = len(pool)

    # ①-③ 基础过滤
    pool = pool[~pool['name'].str.contains('ST|^\*ST', na=False)]
    pool = pool[~pool['ts_code'].str.endswith('.BJ')]
    pool = pool[~pool['ts_code'].str.match(r'688\d{3}\.SH')]
    print(f"  剔除ST/北交所/科创板: {before} → {len(pool)}")

    # ④ MA多头
    before = len(pool)
    pool = pool[(pool['ma5'] > pool['ma10']) & (pool['ma10'] > pool['ma20']) & (pool['close_end'] > pool['ma20'])].reset_index(drop=True)
    print(f"  MA5>MA10>MA20: {before} → {len(pool)}")

    # ⑤ 成交额TOP500
    before = len(pool)
    pool = pool.sort_values('amount', ascending=False).head(500).reset_index(drop=True)
    print(f"  成交额TOP500: {before} → {len(pool)}")

    # ⑥ 换手率分档
    before = len(pool)
    cond_big = (pool['total_mv_yi'] >= 500) & (pool['turnover_rate'] >= 3)
    cond_mid = (pool['total_mv_yi'] >= 100) & (pool['total_mv_yi'] < 500) & (pool['turnover_rate'] >= 5)
    cond_sml = (pool['total_mv_yi'] < 100) & (pool['turnover_rate'] >= 8)
    pool = pool[cond_big | cond_mid | cond_sml].reset_index(drop=True)
    print(f"  换手率分档: {before} → {len(pool)}")

    # ⑦ 5日均额 > 20日均额
    before = len(pool)
    pool = pool[pool['amt_5d'] > pool['amt_20d']].reset_index(drop=True)
    print(f"  5日均额>20日均额: {before} → {len(pool)}")

    # ⑧ 涨幅>=2% 且 收盘>均价
    before = len(pool)
    pool = pool[(pool['daily_pct'] >= 2) & (pool['close_end'] > pool['avg_price'])].reset_index(drop=True)
    print(f"  涨幅>=2%+收盘>均价: {before} → {len(pool)}")

    # ====== 输出 ======
    codes = pool['ts_code'].str.split('.').str[0].tolist()
    stock_list = ','.join(codes)

    print(f"\n{'='*50}")
    print(f"📊 最终: {len(pool)} 只")
    print(f"{'='*50}")
    for i, s in pool.iterrows():
        print(f"  {s['ts_code'].split('.')[0]:<8} {s['name']:<10} {s['industry']:<8} "
              f"收盘{s['close_end']:<8.2f} 涨幅{s['daily_pct']:<+6.1f}% "
              f"换手{s['turnover_rate']:<6.2f}% PE{s['pe']:<6.0f} "
              f"市值{s['total_mv_yi']:<6.0f}亿")

    print(f"\n📤 STOCK_LIST={stock_list}")
    print(f"   共 {len(codes)} 只")

    # 输出到文件
    os.makedirs("output", exist_ok=True)
    pool[['ts_code','name','industry','close_end','daily_pct','turnover_rate','pe','total_mv_yi',
          'ma5','ma10','ma20','amt_5d','amt_20d','amount_yi']].to_csv("output/screener_result.csv", index=False)
    with open("output/stock_list.env", "w") as f:
        f.write(f"STOCK_LIST={stock_list}")

    return stock_list

if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else os.getenv("TUSHARE_TOKEN", "")
    if not token:
        print("请提供 TUSHARE_TOKEN")
        sys.exit(1)
    run(token)
