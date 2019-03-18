import pandas as pd
import pdb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import os

def get_units(inp):
    cols = inp.columns.values
    which = np.isin(cols,np.array(['trial_number','orientations','contrasts','phases','durations']),invert=True)
    return cols[which]
    
def filter_session(inp,unit_filter='all',trial_filter='all',time_filter=np.array([-np.inf,np.inf])):
    
    if unit_filter=='all':
        unit_filter = get_units(inp)
    elif isinstance(unit_filter,str):
        unit_filter = [unit_filter]
    if trial_filter=='all':
        trial_filter = inp.trial_number
    
    def get_length(ii,min,max):
        ii = ii[np.bitwise_and(ii>min,ii<=max)]
        return ii.size
    
    out = inp.filter(items=['trial_number','orientations','contrasts','phases','durations'])
    for u in unit_filter:
        s_u = inp[u]
        s_u = s_u.apply(get_length,convert_dtype=True,args=(time_filter[0],time_filter[1]))
        out[u] = s_u
    return out

def is_consistent(coeffs,frac=0.7):
    num = coeffs.size
    sign = coeffs/abs(coeffs)
    num_given_sign = num-(num-np.abs(np.sum(sign)))/2
    if np.any(np.isnan(coeffs)): pdb.set_trace()
    if num_given_sign/num>frac: return True
    else: return False

def predict_ori(df,n_splits=100,remove_0_contrast=False,fit_intercept=True,verbose=False):
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


if __name__=='__main__':
    loc = '/camhpc/home/bsriram/data/Analysis/ShortDurSessionDFs'
    save_loc = '/camhpc/home/bsriram/data/Analysis'
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