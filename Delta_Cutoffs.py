#!/usr/bin/env python
# coding: utf-8

## import packages needed
from math import log, sqrt, pi, exp, erf
from scipy.stats import norm
from datetime import datetime, date
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import itertools
import datetime, time
from copy import deepcopy

# Underlying price (say, ETH): S;
# Strike: K;
# Time to maturity (years): T;
# Risk free rate: r;
# Volatility: sigma;


NN = norm.cdf
BASE_CAP = 10
BASE_FLOOR = 0.1
SKEW_CAP = 10
SKEW_FLOOR = 0.1

## define the call/puts price

def d1(sigma, K, S, r, T):
    return (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))


def d2(sigma, K, S, r, T):
    return d1(sigma, K, S, r, T) - sigma * np.sqrt(T)


def bs_call(sigma, K, S, r, T):
    # d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    # d2 = d1 - sigma * np.sqrt(T)
    return S * NN(d1(sigma, K, S, r, T)) - K * np.exp(-r * T) * NN(d2(sigma, K, S, r, T))


def bs_put(sigma, K, S, r, T):
    # d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    # d2 = d1 - sigma* np.sqrt(T)
    return K * np.exp(-r * T) * NN(-d2(sigma, K, S, r, T)) - S * NN(-d1(sigma, K, S, r, T))


def call_delta(sigma, K, S, r, T):
    return norm.cdf(d1(sigma, K, S, r, T))

def put_delta(sigma, K, S, r, T):
    return  call_delta(sigma, K, S, r, T) - 1

def call_gamma(sigma, K, S, r, T):
    return norm.pdf(d1(sigma, K, S, r, T)) / (S * sigma * sqrt(T))


def call_vega(sigma, K, S, r, T):
    return 0.01 * (S * norm.pdf(d1(sigma, K, S, r, T)) * sqrt(T))


def call_theta(sigma, K, S, r, T):
    return 0.01 * (-(S * norm.pdf(d1(sigma, K, S, r, T)) * sigma) / (2 * sqrt(T)) - r * K * exp(-r * T) * norm.cdf(
        d2(sigma, K, S, r, T)))


def call_rho(sigma, K, S, r, T):
    return 0.01 * (K * T * exp(-r * T) * norm.cdf(d2(sigma, K, S, r, T)))


def GWAV(ts, v, t=None):
    t = ts[-1] + 1 if t is None else t  # assume GWAV is queried an instance after the trade 
    assert t > ts[-1]
    T = (t - ts[0]) / 60 / 60
    dt = np.append((ts[1:] - ts[:-1]), np.array([t - ts[-1]])) / 60 / 60
    return np.prod((v**dt))**(1/T)

def GWAV_N(ts, v, length):
    N=length*60*60
    out = []
    gwav = v[0]
    
    if len(v) == 1:
        return [(ts[0], v[0])]
    
    idx_a = 0
    idx_b = 1
    
    t_a = ts[idx_a]
    t_b = ts[idx_b]
    
    while True:
        
        ts_sub = ts[idx_a: idx_b + 1]  # add 1 since we want to include last ts (will be dropped in GWAV)
        v_sub = v[idx_a: idx_b + 1]
        
        if (t_b - t_a) < N:
            ts_sub = np.append(np.array([t_b - N]), ts_sub)
            v_sub = np.append(np.array([v_sub[0]]), v_sub)
        
        gwav = GWAV(ts_sub, v_sub, ts_sub[-1] + 1)
        
        if len(out) == 0:
            out.append((ts[0], gwav))
        
        out.append((ts[idx_b], gwav))  # timestamp used here is the trade timestamp instead of "+1"
        
        idx_b += 1  # append new ts/vol
        if idx_b >= len(ts):
            break

        t_b = ts[idx_b]
        t_a = t_b - N  # move t_a threshold
        idx_a = max((np.argmin(ts < t_a) - 1), 0)  # first idx that belongs to (t_a, t_b) interval

    return out

def GWAV_N2(ts, v, length):
    N=length*60*60
    out = []
    gwav = v[0]
    
    if len(v) == 1:
        return [ v[0]]
    
    idx_a = 0
    idx_b = 1
    
    t_a = ts[idx_a]
    t_b = ts[idx_b]
    
    while True:
        
        ts_sub = ts[idx_a: idx_b + 1]  # add 1 since we want to include last ts (will be dropped in GWAV)
        v_sub = v[idx_a: idx_b + 1]
        
        if (t_b - t_a) < N:
            ts_sub = np.append(np.array([t_b - N]), ts_sub)
            v_sub = np.append(np.array([v_sub[0]]), v_sub)
        
        gwav = GWAV(ts_sub, v_sub, ts_sub[-1] + 1)
        
        if len(out) == 0:
            out.append(gwav)
        
        out.append(gwav)  # timestamp used here is the trade timestamp instead of "+1"
        
        idx_b += 1  # append new ts/vol
        if idx_b >= len(ts):
            break

        t_b = ts[idx_b]
        t_a = t_b - N  # move t_a threshold
        idx_a = max((np.argmin(ts < t_a) - 1), 0)  # first idx that belongs to (t_a, t_b) interval

    return out

from collections import defaultdict
class AMMPortfolio:
    def __init__(self):
        self.positions = defaultdict(float)
        self.timestamp = 0
        self.price = 0
        self.listings = {}
        
        
    def process_trade(self, ts, d):
        assert ts >= self.timestamp
        trade_type = 'CALL' if 'CALL' in d['tradeType'] else 'PUT'
        direction = 'LONG' if 'LONG' in d['tradeType'] else 'SHORT'
        is_buy = ((d['OpenClose'] == 'Open') and (direction == 'LONG')) or ((d['OpenClose'] == 'Close') and (direction == 'SHORT'))
        
        listing = (d['expiry'], d['strike'], trade_type)
        self.listings[listing] = d
        for k,v in self.listings.items():
            if v['expiry'] == d['expiry']:
                # update same expiry's base IV and trade vol
                self.listings[k]['baseIvGWAV'] = d['baseIvGWAV']
                self.listings[k]['tradeIvGWAV'] = self.listings[k]['baseIvGWAV'] * self.listings[k]['skewGWAV']
                
                self.listings[k]['baseIv'] = d['baseIv']
                self.listings[k]['tradeIv'] = self.listings[k]['baseIv'] * self.listings[k]['skew']
            if v['expiry'] < ts:
                # remove old listings
                self.positions[k] = 0
            
        if is_buy:
            self.positions[listing] -= d['amount']
        else:
            self.positions[listing] += d['amount']
        
        self.timestamp = ts
        self.price = d['price']
    
    def get_net_vega(self, method='tradeIvGWAV'):
        vega = defaultdict(float)
        for k,v in self.listings.items():
            amt = self.positions[k]
            if amt != 0:
                T = (v['expiry'] - self.timestamp) / 60 / 60 / 24 / 365
                vg = call_vega(v[method], v['strike'], self.price, 0.1, T)
                vega[k] += vg * amt
        return vega
    
    def get_nav_diff(self, method='tradeIvGWAV'):
        vega = self.get_net_vega(method)
        diff = 0
        for k,v in self.listings.items():
            iv_diff = (v[method] - v['tradeIv'])
            iv_diff = 0 if abs(iv_diff) > 0.1 else iv_diff  # CB fire
            l_diff = iv_diff * vega[k] * 100
            diff += l_diff
        return diff
    
    
    def get_net_vega(self, method='tradeIvGWAV'):
        vega = self.get_net_vega(method)
        vega_tot = 0
        for k,v in self.listings.items():
            vega_tot += vega[k] * 100
        return diff
    
    
def DataGenerator(length, market='sETH'):
    full_df = pd.read_csv(f"{market}.csv")
    df = full_df
    ts_series = deepcopy(df['timestamp'])
    last_ts = 0
    same_ts_set = set()
    for i, (k,v) in enumerate(df[['block', 'timestamp']].iterrows()):
        """
        Optimism block timestamp is crap, it relates to the block of ETH at which the sync happened, and not the 
        timestapmp of the actual transaction. This means that several txs can have the same timestamp, messing up our
        time series. In order to decouple those transactions from one another, we look for all trades with the same
        timestamp, and at their block numbers (which are unique per trade). We then reset the timestamp of the trade to
        an interpolated value, where the x-axis is the block number and y-axis is the timestamp. For example,
        if we have 4 trades with blocks [1,2,3,4] and timestamps [10,10,10,40], then timestamps of blocks
        2 and 3 will be interpolated, resulting in [10,20,30,40]
        """
        ts = v['timestamp']
        if ts == last_ts:
            same_ts_set.add(i)
            same_ts_set.add(i-1)
        else:
            if len(same_ts_set) > 0:
                same_ts_list = sorted(list(same_ts_set))
                min_b = df.iloc[same_ts_list[0]]['block']
                max_b = df.iloc[i]['block']
            
                min_t = last_ts
                max_t = ts
            
                for j in range(1, len(same_ts_list)):
                    n = same_ts_list[j]
                    b = df.iloc[n]['block']
                    assert b != min_b
                    assert b != max_b
                    assert max_t > min_t
                    assert b > min_b
                    assert b < max_b
                    interp_t = np.interp(b, [min_b, max_b], [min_t, max_t])
                    ts_series.iloc[n] = round(interp_t)
            same_ts_set = set()
            last_ts = ts
    
    full_df = full_df.sort_values(by=['block']).set_index('timestamp')
    
    #last_round = sorted(full_df.expiry.unique())[-4:] #uncomment to retrieve the orginal
    last_round = sorted(full_df.expiry.unique())[-14:]

   #print(last_round)
    
    if length == 0:
        return full_df
    
    lr_df = deepcopy(full_df[full_df.expiry.isin(last_round)])
    lr_df['baseIvGWAV'] = np.nan
    lr_df['skewGWAV'] = np.nan
    combs = []

    for m in lr_df.expiry.unique():
        combs.append((m, lr_df[lr_df.expiry == m].strike.unique()))
    
    for m, ks in combs:
        mask = (lr_df.expiry == m)
        gwavs = GWAV_N(lr_df[mask].index.values,
                   lr_df[mask]['baseIv'].values,
                   length)
        gwav_df = pd.DataFrame(gwavs, columns=['timestamp', 'baseIvGWAV']).set_index('timestamp')
        lr_df.loc[gwav_df.index, 'baseIvGWAV'] = gwav_df.loc[gwav_df.index, 'baseIvGWAV']
    
        for k in ks:
            mask = (lr_df.expiry == m) & (lr_df.strike == k) 
            gwavs = GWAV_N(lr_df[mask].index.values,
               lr_df[mask]['skew'].values,
               length)
            gwav_df = pd.DataFrame(gwavs, columns=['timestamp', 'skewGWAV']).set_index('timestamp')
            lr_df.loc[gwav_df.index, 'skewGWAV'] = gwav_df.loc[gwav_df.index, 'skewGWAV']
        
    lr_df['tradeIvGWAV'] = lr_df['baseIvGWAV'] * lr_df['skewGWAV']
    lr_df['diffGWAV'] = lr_df['tradeIvGWAV'] - lr_df['tradeIv']

    return(lr_df)

def Top_Share(data,share):
    data_length = int(len(data)*share/100.0);
    desired_top_vals = np.sort(data)[-data_length:]
    desired_bottom_vals = np.sort(data)[:data_length]
    return [desired_bottom_vals,desired_top_vals]

def Combined_Data(gma_length):
    
    percent_anal = 10;
    
    data = DataGenerator(gma_length);
    
    diff_tradevol_GMA_all = data['diffGWAV'];
    diff_skew_GMA_all = data['skew'] - data['skewGWAV']
    diff_baseIV_GMA_all = data['baseIv'] - data['baseIvGWAV']
    
    fig, axs = plt.subplots(1, 3,figsize=(18,4));
    axs[0].hist(diff_tradevol_GMA_all, density = True, bins = 50, color = 'gray')
    axs[1].hist(diff_baseIV_GMA_all, density = True, bins = 50, color = 'gray')
    axs[2].hist(diff_skew_GMA_all, density = True, bins = 50, color = 'gray')
    
    mean_tradevol_diff_all = np.mean(diff_tradevol_GMA_all)
    mean_skew_diff_all = np.mean(diff_skew_GMA_all)
    mean_base_diff_all = np.mean(diff_baseIV_GMA_all)
    
    bottom_trade_vol_all, top_trade_vol_all = Top_Share(data['diffGWAV'], percent_anal)
    bottom_trade_vol_all_MEAN = np.mean(bottom_trade_vol_all)
    top_trade_vol_all_MEAN = np.mean(top_trade_vol_all)
    
    bottom_base_all, top_base_all = Top_Share(diff_baseIV_GMA_all, percent_anal)
    bottom_base_all_MEAN = np.mean(bottom_base_all)
    top_base_all_MEAN = np.mean(top_base_all)
    
    bottom_skew_all, top_skew_all = Top_Share(diff_skew_GMA_all, percent_anal)
    bottom_skew_all_MEAN = np.mean(bottom_skew_all)
    top_skew_all_MEAN = np.mean(top_skew_all)
    
    top_means = [top_trade_vol_all_MEAN, top_base_all_MEAN, top_skew_all_MEAN]
    bottom_means = [bottom_trade_vol_all_MEAN, bottom_base_all_MEAN, bottom_skew_all_MEAN]
    
    percent_99 = int(0.99 * len(data))
    
    datas = [data['diffGWAV'], diff_baseIV_GMA_all, diff_skew_GMA_all]
    tops = [top_trade_vol_all, top_base_all, top_skew_all]
    bottoms = [bottom_trade_vol_all, bottom_base_all, bottom_skew_all]
    
    ylims = [20,20,80]
    
    for i in range(0,3):
        axs[i].axvline(datas[i].mean(), color='k', linestyle='dashed', linewidth=1)
        axs[i].set_ylim([0,ylims[i]]) # Heavily truncating the ceiling to get a beter perspective
        axs[i].axvline(tops[i].mean(), color='r', linestyle='dashed', linewidth=1)
        axs[i].axvline(bottoms[i].mean(), color='r', linestyle='dashed', linewidth=1)
        axs[i].axvline(np.sort(datas[i])[percent_99], color = 'b', linestyle = 'dashed', linewidth = 1)
        axs[i].axvline(np.sort(datas[i])[-percent_99], color = 'b', linestyle = 'dashed', linewidth = 1)
        axs[i].set_xlim([np.sort(1.2*datas[i])[-percent_99],1.2*np.sort(datas[i])[percent_99]])
    
    axs[0].set_title("Trading GWAV - Spot")
    axs[1].set_title("BaseIV GWAV - Spot")
    axs[2].set_title("Skew GWAV - Spot")
    
    ret_data = {"GMA Length":gma_length, "TradeVol Metrics": {}, "BaseVol Metrics":{}, "Skew Metrics": {}};
    metrics = ["TradeVol Metrics", "BaseVol Metrics", "Skew Metrics"]
    for i in range(0,3):
        ret_data[metrics[i]]["Overall Average"] = np.mean(datas[i])
        ret_data[metrics[i]]["Overall STD"] = np.std(datas[i])
        ret_data[metrics[i]]["Top 99% Average"] = np.mean(tops[i])
        ret_data[metrics[i]]["Bottom 1% Average"] = np.mean(bottoms[i])
        ret_data[metrics[i]]["Top 99%"] = np.sort(datas[i])[percent_99]
        ret_data[metrics[i]]["Bottom 1%"] = np.sort(datas[i])[-percent_99]
    return ret_data
    
def spot_distributions(data):
    data = DataGenerator(1);
    
    baseIvs = data['baseIv']
    skews = data['skew']
    tradeIvs = data['tradeIv']
    
    bottom_10_base, top_90_base = Top_Share(baseIvs, 10)
    bottom_1_base, top_1_base = Top_Share(baseIvs,1)
    base_extremes = [[bottom_10_base, top_90_base], [bottom_1_base, top_1_base]]
    
    bottom_10_skew, top_90_skew = Top_Share(skews,10)
    bottom_1_skew, top_1_skew = Top_Share(skews,1)
    skew_extremes = [[bottom_10_skew, top_90_skew], [bottom_1_skew, top_1_skew]]
    
    bottom_10_trade, top_90_trade = Top_Share(tradeIvs, 10)
    bottom_1_trade, top_1_trade = Top_Share(tradeIvs,1)
    trade_extremes = [[bottom_10_trade, top_90_trade], [bottom_1_trade, top_1_trade]]
    
    plot_data = [tradeIvs, baseIvs, skews];
    
    extremes = [trade_extremes, base_extremes, skew_extremes]
    
    fig, axs = plt.subplots(1, 3,figsize=(18,4));
    
    axs[0].hist(tradeIvs, bins = 30, color = 'gray')
    axs[1].hist(baseIvs, bins = 30, color = 'gray')
    axs[2].hist(skews, bins = 30, color = 'gray')
    
    axs[0].set_xlabel("tradeIvs")
    axs[1].set_xlabel("baseIvs")
    axs[2].set_xlabel("skews")
    
    axs[0].set_title("TradeIv Distribution")
    axs[1].set_title("BaseIv Distribution")
    axs[2].set_title("SKew Distribution")
    
    for i in range(0,3):
        axs[i].axvline(plot_data[i].mean(), color='k', linestyle='dashed', linewidth=1)
        #axs[i].set_ylim([0,ylims[i]]) # Heavily truncating the ceiling to get a beter perspective
        axs[i].axvline(extremes[i][0][0].mean(), color='r', linestyle='dashed', linewidth=1)
        axs[i].axvline(extremes[i][0][1].mean(), color='r', linestyle='dashed', linewidth=1)
        axs[i].axvline(extremes[i][1][0].mean(), color = 'b', linestyle = 'dashed', linewidth = 1)
        axs[i].axvline(extremes[i][1][1].mean(), color = 'b', linestyle = 'dashed', linewidth = 1)
        #axs[i].set_xlim([np.sort(1.2*datas[i])[-percent_99],1.2*np.sort(datas[i])[percent_99]])
        
        
from datetime import datetime
def AddDateTime(length):
    lr_df = DataGenerator(length)
    
    m = np.sort(list(lr_df.expiry.unique()));
    
    datetest=np.sort(list(lr_df.expiry.unique()));
    dates = [datetime.utcfromtimestamp(list(lr_df.expiry.unique())[i]).strftime('%Y-%m-%d %H:%M:%S') for i in range(0,len(datetest))]
    dates = sorted(dates)
    
    date_df = lr_df['expiry']
    date_df_proper = [datetime.utcfromtimestamp(list(date_df)[i]).strftime('%Y-%m-%d %H:%M:%S') for i in range(0,len(date_df))]
    lr_df['datetime'] = date_df_proper
    return lr_df

def Visualiser(lr_df, expiry):
    m = np.sort(list(lr_df.datetime.unique()))[expiry];
    use_df = pd.DataFrame(lr_df[lr_df.datetime==m]).reset_index();
    times = (use_df['timestamp']-use_df['timestamp'][0])/(60*60*24*7);

    fig, axs = plt.subplots(2, 2,figsize=(14,7));
    fig.suptitle("Expiry = {sc}".format(sc = m,fontsize = 16))
    
    axs[0,0].hist(use_df['diffGWAV'],bins = 50, density = True,color = 'blue');
    axs[0,0].title.set_text('(GWAV - Spot) Vol')
    
    axs[0,1].hist(100*(use_df['baseIv']-use_df['baseIvGWAV']),bins=50, density =True,color = 'red');
    axs[0,1].title.set_text('(baseIV - GWAVIV)')
    
    axs[0,0].title.set_text('TradeIV, GWAVIV')
    
    #axs[1,0].plot(times, use_df['tradeIv']);
    #axs[1,0].plot(times, use_df['tradeIvGWAV']);
    #axs[1,0].set_xlabel("Time (weeks)")
    #axs[1,0].set_ylabel("trade BaseIV/GWAV")
    
    axs[1,0].hist(use_df['skew'] - use_df['skewGWAV'], bins = 30, density =True, color = 'green')
    axs[1,0].set_xlabel("Time (weeks)")
    axs[1,0].set_ylabel("Skew spot v GWAV")
    axs[1,0].title.set_text("skew spot, GWAV")
    
    axs[1,1].title.set_text('baseIV, baseGWAV')
    axs[1,1].plot(times, use_df['baseIv']);
    axs[1,1].plot(times, use_df['baseIvGWAV']);
    axs[1,1].set_xlabel("Time (weeks)")
    axs[1,1].set_ylabel("BaseIV/GWAP")

def Skew_Visualiser(lr_df, expiry):
    m = lr_df.expiry.unique()[expiry];
    threshold = 1;
    use_df = pd.DataFrame(lr_df[lr_df.expiry==m]).reset_index();
    fil1= use_df.groupby(["strike"])['trader'].count().reset_index("strike")
    strikes = np.sort(np.array(fil1[fil1.trader > threshold]['strike']));
    
    fig, axs = plt.subplots(len(strikes), 2,figsize=(12,10));
    
    tot_skew_deviations = [];
    
    for i in range(0,len(strikes)):
        strike = strikes[i];
        new_df = use_df[use_df.strike == strike];
        new_df = new_df.reset_index()
        times = (new_df['timestamp'] - new_df['timestamp'][0])/(60*60*24*7);

        axs[i,0].plot(times,new_df['skew'],color = 'blue');
        axs[i,0].plot(times,new_df['skewGWAV'], color = 'red');
        
        axs[i,0].set_xlabel("Time (weeks)")
        axs[i,0].title.set_text('Skew spot and GWAV. K = %i' %strike)
        
        axs[i,1].hist(new_df['skew']-new_df['skewGWAV'],color = 'blue',density = True);
        axs[i,1].axvline((new_df['skew']-new_df['skewGWAV']).mean(), color='k', linestyle='dashed', linewidth=1)
        axs[i,1].title.set_text('spot-GWAV skew. K = %i' %strike)
        
        xx=np.array(new_df['skew']-new_df['skewGWAV']);
        #tot_skew_deviations.flatten();
    fig.tight_layout()
    return xx

def lockout_times(gma_length,lockout_pars):
    lockout_scale_time = 2.0;
    un_traded_assumption = 6.0*60*60;
    
    df = DataGenerator(gma_length).reset_index();
    starttime = df.iloc[0]['timestamp'];
    endtime = df.iloc[-1]['timestamp'];
    total_time = (endtime - starttime)/3600;
    cb = 0
    firing = 0;
    lockout = False;

    lockout_timer = 60*60*lockout_scale_time*gma_length; #This is how long after the CB stops firing will the lockout persist (2xlength of gwav)

    good_times = [starttime]; 
    bad_times =[]; #count the intervals over which deposits are locked out
    times_CB_fires = [] #count the number of instances the CB fires
    failed_conds= []
    
    for i in range(0,len(df)): #loop over every trade over all data
        trade = df.iloc[i];
        time_ellapsed = df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp'] #find time since last trade
    
        tradeIV = trade['tradeIv']
        tradeIVGWAV = trade['tradeIvGWAV']
        baseIV = trade['baseIv']
        baseIVGWAV = trade['baseIvGWAV']
    
        skew = trade['skew'];
        skewGWAV = trade['skewGWAV']
    
        firing = max(0, firing - time_ellapsed)
    
        CB_cond = abs(baseIVGWAV - baseIV)> lockout_pars[0] or abs(tradeIV - tradeIVGWAV)> lockout_pars[2] or abs(skew-skewGWAV) > lockout_pars[1];
    
    #TEST: test the CB condition. Pretty tight bounds on base/trade/skews right now. Can make this more generous
    
        if CB_cond == True:
            failed_conds.append([abs(baseIVGWAV - baseIV),abs(tradeIV - tradeIVGWAV),abs(skew-skewGWAV) ])
            times_CB_fires.append(df.iloc[i]['timestamp']) 
            if lockout == False:
            #if the CB fires and there's no lockout in place, initiate a lockout.
            # set an end date of current time + lockout_timer + time greek cache will take to move gma --> spot if no trades
            # this is the beginnning of a lockout interval, so add to bad_times
                lockout = True;
                poss_end_date = df.iloc[i]['timestamp'] + lockout_timer + un_traded_assumption
                bad_times.append(df.iloc[i]['timestamp'])
            elif lockout == True:
                poss_end_date = df.iloc[i]['timestamp'] + lockout_timer + un_traded_assumption
            #if there's already a lockout in place, just add 
            #+ lockout_timer + time greek cache will take to move gma --> spot if no trades
    
            
        if CB_cond == False:
            if lockout == True and df.iloc[i]['timestamp'] > poss_end_date:
                lockout = False
                bad_times.append(poss_end_date)
            #if CB not firing but there's a lockout and current time > poss_end_date, then the lockout has ended
            # so add poss_end_date to the end of the lockout interval
            elif lockout == False:
                good_times.append(df.iloc[i]['timestamp'])
        
            #if everything good, there's no lockout
    return([good_times, bad_times, times_CB_fires,failed_conds,total_time])

def bad_maker(bad_times):
    bad_list = [] 
    bad_interval_lengths = [];
    
    if len(bad_times)%2 !=0:
        final = len(bad_times)-1
    else: 
        final = len(bad_times)

    for i in range(0,final,2):
        add_bad = (int(bad_times[i]), int(bad_times[i+1]));
        add_bad_length = bad_times[i+1] - bad_times[i];
    
        bad_list.append(add_bad)
        bad_interval_lengths.append(add_bad_length/3600)
    return([bad_list, bad_interval_lengths])

def deposit_times(gma_length, frequency, processing_time):
    deposit_dates = [] #seed a deposit every 1 hour since the start of the dataset. 
    time_12_hrs = 60*60*frequency
    
    
    df = DataGenerator(gma_length).reset_index();
    starttime = df.iloc[0]['timestamp'];
    endtime = df.iloc[-1]['timestamp'];
    
    time = starttime 

    i = 0;

    while time < endtime - time_12_hrs:
        time = starttime + i*time_12_hrs
        i+= 1
        deposit_dates.append(time)
    return [endtime - starttime, deposit_dates]

def delay_finder(bad_list, deposit_dates):
    processing_delay = [];
# for each seeded deposit, if the CB is firing when their deposit is meant to be processed
# process the deposit ad the end of the interval. Otherwise immediately process
    for deposit in deposit_dates:
        print(100*deposit_dates.index(deposit)/len(deposit_dates))
        isdelayed = [deposit in range(r[0],r[1]) for r in bad_list]

        if any(isdelayed) == False:
            processing_delay.append(0)
        
        elif any(isdelayed) == True:
            which_index = isdelayed.index(True)
            which_time = bad_list[which_index][1]
            delay = which_time - (deposit) 
            processing_delay.append(delay)
            
            
    return processing_delay

def percent_lockedout(gma_length, firing_conds):
    goods, bads, firing_times, failed_conds, total_times = lockout_times(gma_length, firing_conds)

    bad_list, bad_interval_lengths = bad_maker(bads)
    
    if len(bad_list) ==0: 
        return 0, 0 ,0

    else:
        percent_time_lockout = 100*sum(bad_interval_lengths)/total_times
    
        return percent_time_lockout, len(bad_list), np.mean(bad_interval_lengths)
    
def flip_amount_on_condition(df):
    if (df['OpenClose'] == "Open") & (df['tradeType'] == "LONG_CALL"):
        return -df['amount']
    
    elif (df['OpenClose'] == "Open") & (df['tradeType'] == "LONG_PUT"):
        return -df['amount']
    
    elif (df['OpenClose'] == "Close") & (df['tradeType'] == "SHORT_CALL"):
        return -df['amount']
    
    elif (df['OpenClose'] == "Close") & (df['tradeType'] == "SHORT_PUT"):
        return -df['amount']
    
    else:
        return df['amount']

def flip_cost_on_condition(df):
    if (df['OpenClose'] == "Open") & (df['tradeType'] == "SHORT_CALL"):
        return -df['totalCost']
    
    elif (df['OpenClose'] == "Open") & (df['tradeType'] == "SHORT_PUT"):
        return -df['totalCost']
    
    elif (df['OpenClose'] == "Close" & df['tradeType'] == "LONG_CALL"):
        return -df['totalCost']
    
    elif (df['OpenClose'] == "Close" & df['tradeType'] == "LONG_PUT"):
        return -df['totalCost']
    
    return df['totalCost']

def find_spot(exps, df):
    spots = {};
    
    for i in range(0,len(exps)):
        
        actual_time = np.sort(exps)[i]
        LIST = df['unix']
        closest_time = min(LIST, key=lambda x:abs(x-actual_time))
        spot_val = df[df['unix'] == closest_time]['close']
        spots[str(exps[i])] = float(spot_val)
    return spots

def pnl_calc(amount, spot, strike, tradetype):
    if tradetype == "LONG_CALL" or tradetype == "SHORT_CALL":
        pnl = amount * max(0, spot - strike)
    if tradetype == "LONG_PUT" or tradetype == "SHORT_PUT":
        pnl = amount * max(0, strike - spot)
    return pnl

def trade_delta(df):
    if (df['OpenClose'] == "Open"):
        sign = +1
    if (df['OpenClose'] == "Close"):
        sign = -1
    if (df['tradeType'] == "LONG_CALL"):
        return -sign*df['amount'] * df['callDelta']
    if (df['tradeType'] == "SHORT_CALL"):
        return sign*df['amount'] * df['callDelta']
    if (df['tradeType'] == "LONG_PUT"):
        return -sign*df['amount'] * df['putDelta']
    if (df['tradeType'] == "SHORT_PUT"):
        return sign*df['amount'] * df['putDelta']
    
    