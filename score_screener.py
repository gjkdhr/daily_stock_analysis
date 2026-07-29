"""
加权评分选股模型

指标            权重   数据来源
趋势(MA)        20    MA5/MA10/MA20 排列 + 角度
成交额           15    今日成交额 / 20日均额
资金流           20    收盘价与均价差 × 成交量 (近似)
RS相对强度       15    个股涨幅 vs 沪深300涨幅
板块强度         15    所属行业平均涨幅排名
BIAS健康度       10    BIAS20 乖离率评分
突破形态          5    接近20日高点 + 放量
─────────────────
总分            100
"""
import tushare as ts
import pandas as pd
import os, sys
import numpy as np

def run(token):
    ts.set_token(token)
    pro = ts.pro_api()

    # 交易日历
    cal = pro.trade_cal(start_date="20260601", end_date="20260729")
    days = cal[cal['is_open']==1]['cal_date'].sort_values().tolist()
    end, start = days[-1], days[-20]

    # 股票基础信息
    csv_path = "data/stock_list_a.csv"
    if os.path.exists(csv_path):
        sb = pd.read_csv(csv_path, usecols=['ts_code','name','industry'])
    else:
        sb = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
        sb.to_csv("data/stock_list_a.csv", index=False)

    # 下载数据
    dl, bb = [], []
    for dt in days[-20:]:
        d = pro.daily(trade_date=dt)
        if d is not None and len(d)>0: d['trade_date']=str(dt); dl.append(d)
        b = pro.daily_basic(trade_date=dt)
        if b is not None and len(b)>0: b['trade_date']=str(dt); bb.append(b)
    daily = pd.concat(dl, ignore_index=True)
    basic = pd.concat(bb, ignore_index=True)

    # 沪深300
    csi = pro.index_daily(ts_code='000300.SH', start_date=start, end_date=end)
    csi_s = csi[csi['trade_date']==start]['close'].iloc[0]
    csi_e = csi[csi['trade_date']==end]['close'].iloc[0]
    csi_pct = (csi_e - csi_s) / csi_s * 100

    # 构建池（基础过滤）
    s = daily[daily['trade_date']==start][['ts_code','close']].rename(columns={'close':'cs'})
    e = daily[daily['trade_date']==end][['ts_code','close']].rename(columns={'close':'ce'})
    pool = pd.merge(s, e, on='ts_code').dropna()
    pool = pd.merge(pool, sb, on='ts_code')

    # 过滤 ST/北交所/科创板
    pool = pool[~pool['name'].str.contains('ST|\\*ST',na=False)]
    pool = pool[~pool['ts_code'].str.endswith('.BJ')]
    pool = pool[~pool['ts_code'].str.match(r'688\d{3}\.SH')]

    # 合并行情数据
    for df in [
        daily[daily['trade_date']==end][['ts_code','pct_chg','amount','vol','high','low','open']].rename(columns={'pct_chg':'dp'}),
        basic[basic['trade_date']==end][['ts_code','turnover_rate','pe','pb','total_mv']]
    ]:
        pool = pd.merge(pool, df, on='ts_code', how='left')

    pool['mv'] = pool['total_mv']/10000
    pool['amt_yi'] = pool['amount']/100000
    pool['avg_p'] = pool['amount']*10/pool['vol']

    # 计算MA/量能指标
    def cma(g,n): return g.sort_values('trade_date',ascending=False).head(n)['close'].mean()
    ma = daily.groupby('ts_code').apply(lambda g: pd.Series({
        'ma5':cma(g,5),'ma10':cma(g,10),'ma20':cma(g,20),
        'high20':g.sort_values('trade_date',ascending=False).head(20)['high'].max()
    })).reset_index()
    def camt(g,n): return g.sort_values('trade_date',ascending=False).head(n)['amount'].mean()
    pool = pd.merge(pool, ma, on='ts_code')
    pool = pd.merge(pool, daily.groupby('ts_code').apply(lambda g: camt(g,5)).reset_index(name='a5'), on='ts_code')
    pool = pd.merge(pool, daily.groupby('ts_code')['amount'].mean().reset_index(name='a20'), on='ts_code')
    pool['amt_ratio'] = pool['amount'] / pool['a20']  # 额量比

    # 个股近20日涨幅 & 超额收益
    pool['pct_20d'] = round((pool['ce'] - pool['cs']) / pool['cs'] * 100, 2)
    pool['excess'] = pool['pct_20d'] - csi_pct
    pool['bias20'] = round((pool['ce'] - pool['ma20']) / pool['ma20'] * 100, 2)

    # ─── 前置筛选（同 ci_screener）───
    print(f"\n📋 前置筛选:")
    before = len(pool)
    pool = pool[(pool['ma5']>pool['ma10'])&(pool['ma10']>pool['ma20'])&(pool['ce']>pool['ma20'])]
    pool = pool.sort_values('amount',ascending=False).head(500)
    c_b=(pool['mv']>=500)&(pool['turnover_rate']>=3); c_m=(pool['mv']>=100)&(pool['mv']<500)&(pool['turnover_rate']>=5); c_s=(pool['mv']<100)&(pool['turnover_rate']>=8)
    pool = pool[c_b|c_m|c_s]
    pool = pool[pool['a5']>pool['a20']]
    pool = pool[(pool['dp']>=2)&(pool['ce']>pool['avg_p'])].reset_index(drop=True)
    pool = pool[pool['excess'] > 10].reset_index(drop=True)
    pool['bias20'] = round((pool['ce'] - pool['ma20']) / pool['ma20'] * 100, 2)
    pool = pool[(pool['bias20'] >= 5) & (pool['bias20'] <= 18)].reset_index(drop=True)
    print(f"  筛选后: {len(pool)} 只 (剔除 {before - len(pool)} 只)\n")

    # ─── 评分 ───
    def score_trend(row):
        """趋势(MA) 0-20分"""
        s = 0
        if row['ma5'] > row['ma10'] > row['ma20']: s += 10
        elif row['ma5'] > row['ma10']: s += 5
        # MA角度（5日均线上升斜率）
        slope = (row['ma5'] - row['ma20']) / row['ma20'] * 100
        if slope > 5: s += 5
        elif slope > 2: s += 3
        elif slope > 0: s += 1
        # 收盘在MA20之上
        if row['ce'] > row['ma20']: s += 5
        return min(s, 20)

    def score_amount(row):
        """成交额 0-15分"""
        r = row['amt_ratio']
        if r > 3: return 15
        if r > 2: return 12
        if r > 1.5: return 9
        if r > 1: return 6
        if r > 0.5: return 3
        return 0

    def score_moneyflow(row):
        """资金流 0-20分（收盘价 vs 均价 × 成交量的方向）"""
        # 资金流向强度 = (收盘 - 均价) / 均价 * 100
        flow = (row['ce'] - row['avg_p']) / row['avg_p'] * 100
        if flow > 2: return 20
        if flow > 1: return 16
        if flow > 0.5: return 12
        if flow > 0: return 8
        if flow > -0.5: return 4
        return 0

    def score_rs(row):
        """RS相对强度 0-15分"""
        e = row['excess']
        if e > 30: return 15
        if e > 20: return 12
        if e > 15: return 9
        if e > 10: return 6
        if e > 5: return 3
        return 0

    def score_bias(row):
        """BIAS健康度 0-10分"""
        b = abs(row['bias20'])
        if 8 <= b <= 15: return 10     # 最佳区间
        if 5 <= b <= 18: return 8      # 合理区间
        if 3 <= b <= 25: return 5
        return 2

    def score_breakout(row):
        """突破形态 0-5分"""
        s = 0
        # 接近20日高点
        if row['ce'] >= row['high20'] * 0.97: s += 3
        elif row['ce'] >= row['high20'] * 0.95: s += 1
        # 放量
        if row['amt_ratio'] > 1.5: s += 2
        elif row['amt_ratio'] > 1.2: s += 1
        return min(s, 5)

    # 板块强度（行业平均涨幅排名）
    ind_pct = pool.groupby('industry')['pct_20d'].mean().rank(pct=True).reset_index()
    ind_pct.columns = ['industry', 'ind_score']
    pool = pd.merge(pool, ind_pct, on='industry')
    pool['score_ind'] = round(pool['ind_score'] * 15, 1)  # 0-15分

    # 计算总分
    pool['score_trend'] = pool.apply(score_trend, axis=1)
    pool['score_amount'] = pool.apply(score_amount, axis=1)
    pool['score_mf'] = pool.apply(score_moneyflow, axis=1)
    pool['score_rs'] = pool.apply(score_rs, axis=1)
    pool['score_bias'] = pool.apply(score_bias, axis=1)
    pool['score_bo'] = pool.apply(score_breakout, axis=1)
    pool['total_score'] = (pool['score_trend'] + pool['score_amount'] + pool['score_mf'] +
                           pool['score_rs'] + pool['score_ind'] + pool['score_bias'] + pool['score_bo'])

    # 按总分排序取前30
    pool = pool.sort_values('total_score', ascending=False).head(30).reset_index(drop=True)

    # 输出
    print(f"\n{'='*90}")
    print(f"📊 评分选股结果 (沪深300近20日: {csi_pct:+.2f}%)")
    print(f"{'='*90}")
    print(f"{'排名':<4} {'代码':<10} {'名称':<8} {'总分':<6} {'趋势':<5} {'额':<4} {'资金':<5} {'RS':<4} {'板块':<5} {'BIAS':<5} {'突破':<4}")
    print(f"{'-'*65}")
    for i, r in pool.iterrows():
        scores = [r['score_trend'],r['score_amount'],r['score_mf'],r['score_rs'],
                  r['score_ind'],r['score_bias'],r['score_bo']]
        s_str = " ".join(f"{s:<4}" for s in scores)
        print(f"{i+1:<4} {r['ts_code'].split('.')[0]:<10} {r['name']:<8} {r['total_score']:<6.0f} {s_str}")

    print(f"\n📤 STOCK_LIST={','.join(pool['ts_code'].str.split('.').str[0].tolist())}")
    return pool

if __name__=="__main__":
    t = sys.argv[1] if len(sys.argv)>1 else os.getenv("TUSHARE_TOKEN","")
    if not t: print("请提供 TUSHARE_TOKEN"); sys.exit(1)
    run(t)
