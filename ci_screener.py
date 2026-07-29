"""优化版：独立筛选脚本
- 股票基础信息读取 DSA 本地 CSV（省 1 次 API）
- 仅需 20 次日线 + 20 日指标 = 40 次调用
- 输出 STOCK_LIST 格式，兼容 DSA
"""
import tushare as ts
import pandas as pd
import os, sys

def run(token):
    ts.set_token(token)
    pro = ts.pro_api()

    # 交易日历
    cal = pro.trade_cal(start_date="20260601", end_date="20260729")
    days = cal[cal['is_open']==1]['cal_date'].sort_values().tolist()
    end, start = days[-1], days[-20]

    # 股票基础信息（优先读 DSA 本地 CSV，省 1 次 API）
    csv_path = "data/stock_list_a.csv"
    if os.path.exists(csv_path):
        sb = pd.read_csv(csv_path, usecols=['ts_code','name','industry'])
        print(f"📂 从 {csv_path} 读取股票信息")
    else:
        sb = pro.stock_basic(exchange='', list_status='L',
                             fields='ts_code,name,industry')
        sb.to_csv("data/stock_list_a.csv", index=False)
        print(f"📥 从 API 获取股票信息 ({len(sb)} 只)")

    # 下载 20 天日线 + 指标（2×20=40 次调用）
    dl, bb = [], []
    for dt in days[-20:]:
        d = pro.daily(trade_date=dt)
        if d is not None and len(d)>0:
            d['trade_date']=str(dt); dl.append(d)
        b = pro.daily_basic(trade_date=dt)
        if b is not None and len(b)>0:
            b['trade_date']=str(dt); bb.append(b)
    daily = pd.concat(dl, ignore_index=True)
    basic = pd.concat(bb, ignore_index=True)

    # 构建池
    s = daily[daily['trade_date']==start][['ts_code','close']].rename(columns={'close':'cs'})
    e = daily[daily['trade_date']==end][['ts_code','close']].rename(columns={'close':'ce'})
    pool = pd.merge(s, e, on='ts_code').dropna()
    pool = pd.merge(pool, sb, on='ts_code')

    # 当日涨跌 + 估值 + 成交额
    for df, cols in [
        (daily[daily['trade_date']==end][['ts_code','pct_chg']].rename(columns={'pct_chg':'dp'}), ['ts_code','dp']),
        (basic[basic['trade_date']==end][['ts_code','turnover_rate','pe','pb','total_mv']], ['ts_code','turnover_rate','pe','pb','total_mv']),
        (daily[daily['trade_date']==end][['ts_code','amount','vol']], ['ts_code','amount','vol'])
    ]:
        pool = pd.merge(pool, df, on='ts_code', how='left')
    pool['mv'] = pool['total_mv']/10000
    pool['amt'] = pool['amount']/100000
    pool['avg_p'] = pool['amount']*10/pool['vol']

    # MA5/10/20 + 5日均额/20日均额
    def cma(g,n): return g.sort_values('trade_date',ascending=False).head(n)['close'].mean()
    ma = daily.groupby('ts_code').apply(lambda g: pd.Series({'ma5':cma(g,5),'ma10':cma(g,10),'ma20':cma(g,20)})).reset_index()
    def camt(g,n): return g.sort_values('trade_date',ascending=False).head(n)['amount'].mean()
    for df in [ma,
               daily.groupby('ts_code').apply(lambda g: camt(g,5)).reset_index(name='a5'),
               daily.groupby('ts_code')['amount'].mean().reset_index(name='a20')]:
        pool = pd.merge(pool, df, on='ts_code')

    # ====== 筛选 ======
    c = ~pool['name'].str.contains('ST|\\*ST',na=False) & ~pool['ts_code'].str.endswith('.BJ') & ~pool['ts_code'].str.match(r'688\d{3}\.SH')
    pool = pool[c]
    pool = pool[(pool['ma5']>pool['ma10'])&(pool['ma10']>pool['ma20'])&(pool['ce']>pool['ma20'])]
    pool = pool.sort_values('amount',ascending=False).head(500)
    c_b=(pool['mv']>=500)&(pool['turnover_rate']>=3); c_m=(pool['mv']>=100)&(pool['mv']<500)&(pool['turnover_rate']>=5); c_s=(pool['mv']<100)&(pool['turnover_rate']>=8)
    pool = pool[c_b|c_m|c_s]
    pool = pool[pool['a5']>pool['a20']]
    pool = pool[(pool['dp']>=2)&(pool['ce']>pool['avg_p'])].reset_index(drop=True)

    # 按成交额排序取 TOP 15（流动性最好的优先）
    pool = pool.sort_values('amount', ascending=False).head(15).reset_index(drop=True)

    # 输出
    codes = pool['ts_code'].str.split('.').str[0].tolist()
    stock_list = ','.join(codes)
    print(f"\n最终: {len(pool)} 只")
    for _, r in pool.iterrows():
        print(f"  {r['ts_code'].split('.')[0]:<8} {r['name']:<10} {r['dp']:<+6.1f}% 换手{r['turnover_rate']:<6.2f}% PE{r['pe']:<6.0f} 市值{r['mv']:<6.0f}亿")
    print(f"\nSTOCK_LIST={stock_list}")

    os.makedirs("output", exist_ok=True)
    with open("output/stock_list.env","w") as f: f.write(f"STOCK_LIST={stock_list}")
    return stock_list

if __name__=="__main__":
    t = sys.argv[1] if len(sys.argv)>1 else os.getenv("TUSHARE_TOKEN","")
    if not t: print("请提供 TUSHARE_TOKEN"); sys.exit(1)
    run(t)
