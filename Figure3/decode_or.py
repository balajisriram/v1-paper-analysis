import pandas as pd
import pdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import os
import statsmodels.api as sm
from tqdm import tqdm
import pickle
import sys
import warnings
import random

warnings.simplefilter('error', UserWarning)

def get_units(inp):
    cols = inp.columns.values
    which = np.isin(cols,np.array(['trial_number','orientations','contrasts','phases','durations','pupil_good','running_good','pupil_size','running_speed','trial_time','aroused']),invert=True)
    return cols[which]
    
def filter_session(inp,unit_filter='all',trial_filter='all',time_filter=np.array([-np.inf,np.inf]),arousal=None):
    
    if unit_filter=='all':
        unit_filter = get_units(inp)
    elif isinstance(unit_filter,str):
        unit_filter = [unit_filter]
    if trial_filter=='all':
        trial_filter = inp.trial_number
    
    def get_length(ii,min,max):
        ii = ii[np.bitwise_and(ii>min,ii<=max)]
        return ii.size
    
    out = inp.filter(items=['trial_number','orientations','contrasts','phases','durations','pupil_good','running_good','pupil_size','running_speed','trial_time','aroused'])
    for u in unit_filter:
        s_u = inp[u]
        s_u = s_u.apply(get_length,convert_dtype=True,args=(time_filter[0],time_filter[1]))
        out[u] = s_u
        
    if not arousal is None:
        if arousal==True:
            out = out.query('aroused==True')
        else:
            out = out.query('aroused==False')
    return out

def is_consistent(coeffs,frac=0.7):
    num = coeffs.size
    sign = coeffs/abs(coeffs)
    num_given_sign = num-(num-np.abs(np.sum(sign)))/2
    if np.any(np.isnan(coeffs)): pdb.set_trace()
    if num_given_sign/num>frac: return True
    else: return False
    
def is_consistent_sm(pvals,frac=0.7):
    pvals = pvals[~np.isnan(pvals)]
    num = pvals.size
    num_sig = np.sum(pvals<0.05)
    try:
        if num_sig/num>frac: return True
        else: return False
    except Exception as e:
        print(e)
        return False

def predict_ori_sk(df,n_splits=100,remove_0_contrast=False,fit_intercept=True,verbose=False):
    X = df[get_units(df)]
    y = df['orientations']
    num_units = len(get_units(df))
    if verbose:
        print('units found : {0}'.format(get_units(df)))
        print('num units found: {0}'.format(num_units))
        print(df['orientations'].describe())
    # deal with totally bananas orientations
    transforms = {}
    for ori in y.unique():
        mapped_ori = ori
        while mapped_ori>90:
            mapped_ori = mapped_ori-180
        while mapped_ori<-90:
            mapped_ori = mapped_ori+180
        transforms[ori]=mapped_ori
    y = y.map(transforms)
    
    # make the orientations 0 or 1
    transforms = {}
    for ori in y.unique():
        if ori>0:transforms[ori] = 1
        elif ori<0:transforms[ori]=0
        else: transforms[ori]=0
    y = y.map(transforms)
    performance = []
    coeffs = []
    intercepts = []
    for i in range(n_splits):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
        try:
            logreg = LogisticRegression(solver='liblinear',multi_class='auto')
            logreg.fit(X_train,y_train)
            coeffs.append(np.squeeze(logreg.coef_))
            intercepts.append(np.squeeze(logreg.intercept_))
            performance.append(logreg.score(X_test,y_test))
        except ValueError:
            coeffs.append(np.nan)
            intercepts.append(np.nan)
            performance.append(np.nan)
        except Exception as e:
            pdb.set_trace()
    if num_units==1:consistent = is_consistent(np.array(coeffs))
    else: consistent = 'n/a'
    return performance,np.array(coeffs),np.array(intercepts),consistent

def predict_ori_sm(df,n_splits=100,remove_0_contrast=False,fit_intercept=True,verbose=False):
    X = df[get_units(df)]
    y = df['orientations']
    
    ctrs = df.contrasts
    durs = df.durations
    
    # fix durations...
    durs[durs<0.05] = 0.05
    durs[durs>0.2] = 0.2
    
    interested_contrasts = [0,0.15,1]
    interested_durations = [0.05,0.1,0.15,0.2]
    
    num_units = len(get_units(df))
    if verbose:
        print('units found : {0}'.format(get_units(df)))
        print('num units found: {0}'.format(num_units))
        print(df['orientations'].describe())
    # deal with totally bananas orientations
    transforms = {}
    for ori in y.unique():
        mapped_ori = ori
        while mapped_ori>90:
            mapped_ori = mapped_ori-180
        while mapped_ori<-90:
            mapped_ori = mapped_ori+180
        transforms[ori]=mapped_ori
    y = y.map(transforms)
    
    # make the orientations 0 or 1
    transforms = {}
    for ori in y.unique():
        if ori>0:transforms[ori] = 1
        elif ori<0:transforms[ori]=0
        else: transforms[ori]=0
    y = y.map(transforms)
    performance = []
    coeffs = []
    intercepts = []
    pvals = []
    perf_matrix_dur_ctr = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
    for i in range(n_splits):
        print(".",end=" ")
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
        if fit_intercept:
            X_train['intercept'] = 1.0
            X_test['intercept'] = 1.0
        try:
            logreg = sm.Logit(y_train,X_train)
            res = logreg.fit(disp=False)
            predicted = res.predict(X_test)
            predicted = (predicted>=0.5)
            perf = np.sum(predicted==y_test)/y_test.size
            coeffs.append(res.params[0])
            intercepts.append(res.params[1])
            performance.append(perf)
            pvals.append(res.pvalues[0])
            
            # okay now try using that model to predict for all specific orientations and durations
            for ii,ctr in enumerate(interested_contrasts):
                for jj,dur in enumerate(interested_durations):
                    which = np.bitwise_and(ctrs==ctr,durs==dur)
                    if np.sum(which)>0:
                        X_sub = X[which]
                        X_sub['intercept'] = 1.0
                        y_sub = y[which]
                        predicted_sub = res.predict(X_sub)
                        predicted_sub = (predicted_sub>=0.5)
                        perf_sub = np.sum(predicted_sub==y_sub)/y_sub.size
                        perf_matrix_dur_ctr[ii][jj].append(perf_sub)
                    else:
                        perf_matrix_dur_ctr[ii][jj].append(np.nan)
                        
        except Exception as e:
            print('Unknown Error :::::::::',get_units(df),e)
            coeffs.append(np.nan)
            intercepts.append(np.nan)
            performance.append(np.nan)
            pvals.append(np.nan)
            
            # fill'em up with nans
            for ii,ctr in enumerate(interested_contrasts):
                for jj,dur in enumerate(interested_durations):
                    perf_matrix_dur_ctr[ii][jj].append(np.nan)
    print("DONE")
    if num_units==1:consistent = is_consistent_sm(np.array(pvals))
    else: consistent = 'n/a'
    return performance,coeffs,intercepts,pvals,consistent,perf_matrix_dur_ctr

def process_session(loc,df_name,fit_intercept=True):
    df = pd.read_pickle(os.path.join(loc,df_name))
    units = get_units(df)
    units_this_session = []
    for unit in units:
        this_unit = {}
        this_unit['unit_id'] = unit
        df_filt = filter_session(df,unit_filter=unit,time_filter=np.array([0,0.5]))
        prefs,coeffs,intercepts,pvals,consistency,perf_matrix_dur_ctr = predict_ori_sm(df_filt,verbose=False,fit_intercept=fit_intercept)
        this_unit['mean_performance'] = np.nanmean(prefs)
        this_unit['mean_coeff'] = np.nanmean(coeffs)
        this_unit['mean_intercept'] = np.nanmean(intercepts)
        this_unit['is_consistent'] = consistency
        this_unit['perf_matrix_dur_ctr'] = perf_matrix_dur_ctr
        units_this_session.append(this_unit)
    save_loc = '/camhpc/home/bsriram/data/Analysis/TempPerfStore'
    with open(os.path.join(save_loc,df_name),'wb') as f:
        pickle.dump(units_this_session,f)
    with open(os.path.join(save_loc,'Finished_sessions.txt'),'a') as f:
        f.write(df_name+'\n')
    return 0
    
def process_session_arousal(loc,df_name,fit_intercept=True):
    df = pd.read_pickle(os.path.join(loc,df_name))
    units = get_units(df)
    units_this_session = []
    for unit in units:
        print(unit)
        this_unit = {}
        this_unit['unit_id'] = unit
        
        df_filt = filter_session(df,unit_filter=unit,time_filter=np.array([0,0.5]),arousal=True)
        prefs,coeffs,intercepts,pvals,consistency,perf_matrix_dur_ctr = predict_ori_sm(df_filt,verbose=False,fit_intercept=fit_intercept)
        this_unit['mean_performance_aroused'] = np.nanmean(prefs)
        this_unit['mean_coeff_aroused'] = np.nanmean(coeffs)
        this_unit['mean_intercept_aroused'] = np.nanmean(intercepts)
        this_unit['is_consistent_aroused'] = consistency
        this_unit['perf_matrix_dur_ctr_aroused'] = perf_matrix_dur_ctr
        
        df_filt = filter_session(df,unit_filter=unit,time_filter=np.array([0,0.5]),arousal=False)
        prefs,coeffs,intercepts,pvals,consistency,perf_matrix_dur_ctr = predict_ori_sm(df_filt,verbose=False,fit_intercept=fit_intercept)
        this_unit['mean_performance_nonaroused'] = np.nanmean(prefs)
        this_unit['mean_coeff_nonaroused'] = np.nanmean(coeffs)
        this_unit['mean_intercept_nonaroused'] = np.nanmean(intercepts)
        this_unit['is_consistent_nonaroused'] = consistency
        this_unit['perf_matrix_dur_ctr_nonaroused'] = perf_matrix_dur_ctr
        
        
        units_this_session.append(this_unit)
    save_loc = r'C:\Users\bsriram\Desktop\Data_V1Paper\Analysis\StatsmodelsPerformance_forArousal'
    with open(os.path.join(save_loc,df_name),'wb') as f:
        pickle.dump(units_this_session,f)
    with open(os.path.join(save_loc,'Finished_sessions.txt'),'a') as f:
        f.write(df_name+'\n')
    return 0

def predict_ori_sm_full(df,n_splits=100,remove_0_contrast=False,fit_intercept=True,verbose=False):
    X = df[get_units(df)]
    y = df['orientations']
    
    ctrs = df.contrasts
    durs = df.durations
    
    # fix durations...
    durs[durs<0.05] = 0.05
    durs[durs>0.2] = 0.2
    
    interested_contrasts = [0,0.15,1]
    interested_durations = [0.05,0.1,0.15,0.2]
    
    num_units = len(get_units(df))
    if verbose:
        print('units found : {0}'.format(get_units(df)))
        print('num units found: {0}'.format(num_units))
        print(df['orientations'].describe())
    # deal with totally bananas orientations
    transforms = {}
    for ori in y.unique():
        mapped_ori = ori
        while mapped_ori>90:
            mapped_ori = mapped_ori-180
        while mapped_ori<-90:
            mapped_ori = mapped_ori+180
        transforms[ori]=mapped_ori
    y = y.map(transforms)
    
    # make the orientations 0 or 1
    transforms = {}
    for ori in y.unique():
        if ori>0:transforms[ori] = 1
        elif ori<0:transforms[ori]=0
        else: transforms[ori]=0
    y = y.map(transforms)
    performance = []
    perf_matrix_dur_ctr = [[[],[],[],[]],[[],[],[],[]],[[],[],[],[]]]
    for i in range(n_splits):
        print(".", end =" ") 
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)
        if fit_intercept:
            X_train['intercept'] = 1.0
            X_test['intercept'] = 1.0
        try:
            logreg = sm.Logit(y_train,X_train)
            res = logreg.fit(disp=False)
            predicted = res.predict(X_test)
            predicted = (predicted>=0.5)
            perf = np.sum(predicted==y_test)/y_test.size
            performance.append(perf)
            
            # okay now try using that model to predict for all specific orientations and durations
            for ii,ctr in enumerate(interested_contrasts):
                for jj,dur in enumerate(interested_durations):
                    which = np.bitwise_and(ctrs==ctr,durs==dur)
                    if np.sum(which)>0:
                        X_sub = X[which]
                        X_sub['intercept'] = 1.0
                        y_sub = y[which]
                        predicted_sub = res.predict(X_sub)
                        predicted_sub = (predicted_sub>=0.5)
                        perf_sub = np.sum(predicted_sub==y_sub)/y_sub.size
                        perf_matrix_dur_ctr[ii][jj].append(perf_sub)
                    else:
                        perf_matrix_dur_ctr[ii][jj].append(np.nan)
            
        except ValueError:
            performance.append(np.nan)
            # fill'em up with nans
            for ii,ctr in enumerate(interested_contrasts):
                for jj,dur in enumerate(interested_durations):
                    perf_matrix_dur_ctr[ii][jj].append(np.nan)
                        
        except Exception as e:
            print('Unknown Error :::::::::',get_units(df),e)
            performance.append(np.nan)
            # fill'em up with nans
            for ii,ctr in enumerate(interested_contrasts):
                for jj,dur in enumerate(interested_durations):
                    perf_matrix_dur_ctr[ii][jj].append(np.nan)
        except Warning as w:
            print(w)
            pdb.set_trace()
    print("done")
    return performance,perf_matrix_dur_ctr

    
def shuffle_df(df):
    output_df = pd.DataFrame()
    for ii,ori in enumerate(df.orientations.unique()):
        for jj,ctr in enumerate(df.contrasts.unique()):
            for kk,dur in enumerate(df.durations.unique()):
                which = np.bitwise_and(df.orientations==ori,df.contrasts==ctr)
                which = np.bitwise_and(which,df.durations==dur)
                sub_df = df.loc[which]
                if not sub_df.empty:
                    sub_df = sub_df.sample(frac=1).reset_index(drop=True)
                    output_df = output_df.append(sub_df,ignore_index=True)
                del sub_df                    
    return output_df              
    
def process_full_session(loc,df_name,fit_intercept=True,n_shuffles=100):
    df = pd.read_pickle(os.path.join(loc,df_name))
    units = get_units(df)
    
    this_session = {}
    this_session['units'] = units
    df_filt = filter_session(df,unit_filter='all',time_filter=np.array([0,0.5]))
    prefs,perf_matrix_dur_ctr = predict_ori_sm_full(df_filt,verbose=False,fit_intercept=fit_intercept)
    perfs_shuff = []
    for ii in range(n_shuffles):
        df_shuffled = shuffle_df(df_filt)
        perfs_shuffle,perf_matrix_dur_ctr_shuffle = predict_ori_sm_full(df_shuffled,verbose=False,fit_intercept=fit_intercept,n_splits=7)
        perfs_shuff.append(perfs_shuffle)
        
    this_session['performances'] = prefs
    this_session['performances_shuffled'] = perfs_shuff
    # this_session['perf_matrix_dur_ctr'] = perf_matrix_dur_ctr
    # save_loc = '/camhpc/home/bsriram/data/Analysis/TempPerfStore'
    save_loc = r'C:\Users\bsriram\Desktop\Data_V1Paper\Analysis\TempPerfStore'
    with open(os.path.join(save_loc,df_name),'wb') as f:
        pickle.dump(this_session,f)
    with open(os.path.join(save_loc,'Finished_sessions.txt'),'a') as f:
        f.write(df_name+'\n')
    return 0
    
def get_firing_rates_by_arousal(loc,df_name):
    df = pd.read_pickle(os.path.join(loc,df_name))
    units = get_units(df)
    
    units = get_units(df)
    units_this_session = []
    for unit in units:
        this_unit = {}
        this_unit['unit_id'] = unit
        df_filt = filter_session(df,unit_filter=unit,time_filter=np.array([0,0.5]))
        this_unit['firing_rate_aroused'] = df_filt.query('aroused==True')[unit].mean()
        this_unit['firing_rate_nonaroused'] = df_filt.query('aroused==False')[unit].mean()
        units_this_session.append(this_unit)
    return units_this_session
    
if __name__=='__main__':
    # loc = '/camhpc/home/bsriram/data/Analysis/ShortDurSessionDFs'
    loc = r'C:\Users\bsriram\Desktop\Data_V1Paper\Analysis\\ShortDurSessionDFsForEyeTracking'
    
    des =r'C:\Users\bsriram\Desktop\Data_V1Paper\Analysis\StatsmodelsPerformance_0-500ms_forEyeTracking'
    
    # for f in os.listdir(loc):
        # if not f in os.listdir(des):
            # print(f)
    print(sys.argv)
    print(int(sys.argv[1]))
    which = int(sys.argv[1])
    which = which-1
    
    try:
        max = int(sys.argv[2])-1
    except:
        max = which+1
    # pool = mp.Pool(24)
    if False:
        time_filters = [np.array([0.,0.01]),
                        np.array([0.,0.025]),
                        np.array([0.,0.05]),
                        np.array([0.,0.1]),
                        np.array([0.,0.25]),
                        np.array([0.,0.5]),
                        np.array([0.,1.]),
                        np.array([0.,2.5]),
                        ]
        names = ['ShortDurDecodingFrame_10ms.df',
                 'ShortDurDecodingFrame_25ms.df',
                 'ShortDurDecodingFrame_50ms.df',
                 'ShortDurDecodingFrame_100ms.df',
                 'ShortDurDecodingFrame_250ms.df',
                 'ShortDurDecodingFrame_500ms.df',
                 'ShortDurDecodingFrame_1000ms.df',
                 'ShortDurDecodingFrame_2500ms.df',]
        for tf,name in zip(time_filters,names):
            unit_list = []
            for f in os.listdir(loc):
                df = pd.read_pickle(os.path.join(loc,f))
                units = get_units(df)
                
                for unit in units:
                    this_unit = {}
                    this_unit['unit_id'] = unit
                    df_filt = filter_session(df,unit_filter=unit)
                    prefs,coeffs,intercepts,consistency = predict_ori(df_filt,verbose=False)
                    this_unit['mean_performance'] = np.mean(prefs)
                    this_unit['mean_coeff'] = np.mean(coeffs)
                    this_unit['is_consistent'] = consistency
                    unit_list.append(this_unit)
            decoding_df = pd.DataFrame(unit_list)
            decoding_df.to_pickle(os.path.join(save_loc,name))
        
    
    f = os.listdir(loc)
    # print(f[which])
    # for f in os.listdir(loc):
    # process_session(loc,f[which],fit_intercept=True)
    all_units = []
    for ii in range(which,max):
        process_session_arousal(loc,f[ii],fit_intercept=True)
        
    pdb.set_trace()
    #for job in f:
    #    pool.apply_async(process_session,args=(loc,f),callback=collect_result, error_callback=handle_error)
        
    #pool.close()
    #pool.join()
    

    
    