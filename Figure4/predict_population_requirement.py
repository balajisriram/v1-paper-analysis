import pandas as pd
import pickle
import pdb
import os
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import statsmodels
from tqdm import tqdm
import sys

sample_from = np.random.choice


def sample_from_population(df, N_trials,N_units,ctr_idx,dur_idx,):
    remove_list = []
    # make sure that the corresponding df has data
    for idx,row in df.iterrows():
        if row.response_hist[0][ctr_idx][dur_idx].size==0 or row.response_hist[1][ctr_idx][dur_idx].size==0:
            remove_list.append(idx)
    df = df.drop(remove_list)
    if df.count==0:
       return [],[],'no_data'
    sub_df = df.sample(n=N_units,replace=True)
    units_this_sample = sub_df.unit_id
    
    orientations = np.random.choice([0,1],N_trials)
    X = np.full((orientations.size,N_units),np.nan)

    for ii,ori in enumerate(orientations):
        for jj,unit in enumerate(units_this_sample):
            r_hist = sub_df.iloc[jj].response_hist
            relevant_resp = r_hist[ori][ctr_idx][dur_idx]
            try:
                X[ii,jj] = sample_from(relevant_resp)
            except ValueError:
                print('relevant_resp:',relevant_resp)

    return X,orientations,'nominal'


def get_performance_for_virtual_session(total_df,N_samplings,N_trials,N_units,ctr_idx,dur_idx):
    performances_that_condition = []

    for i in tqdm(range(N_samplings)):
        retry = True
        while retry:
            X,y,reason = sample_from_population(total_df,N_trials,N_units,ctr_idx,dur_idx)
            if reason=='no_data':
                performances_that_condition = []
                return performances_that_condition
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

            X_train = np.insert(X_train,0,1.0,axis=1)
            X_test = np.insert(X_test,0,1.0,axis=1)
            try:
                logreg = sm.Logit(y_train,X_train)
                res = logreg.fit(disp=False,maxiter=100)
                predicted = res.predict(X_test)
                predicted = (predicted>=0.5)
                perf = np.sum(predicted==y_test)/y_test.size
                retry = False
            except Exception as e:
                print(e)
                retry = True
                
                
        performances_that_condition.append(perf)
    return performances_that_condition
    
if __name__=='__main__':
    which = int(sys.argv[1])
    which = which-1
    total_df = pd.read_pickle('/camhpc/home/bsriram/v1paper/v1-paper-analysis/Figure4/DecodingOfPopulation_onlyConsistent.df')
    save_loc = '/camhpc/home/bsriram/data/Analysis/TempPerfStore'
    # sample N units and create a session for C = 0.15, dur = 0.1
    potential_n_units = [1,2,3,5,8,10,13,15,18,20,23,25,28,30,32,40,50,64,72,96,108,128,176,256,378,512,756,1024]

    N_units = potential_n_units[which]
    N_trials = 1000
    N_samplings = 1000
    potential_orientations = np.array([-45,45])
    potential_contrasts = np.array([0.15, 1])
    potential_durations = np.array([0.05,0.1,0.15, 0.2])
    interested_durations = np.array([0.05,0.1, 0.2])
    print('running for num population==',N_units)
    file_for_pop_size = 'population_{0}.pickle'.format(N_units)
    data_all_condns = []
    for ii,ctr in enumerate(potential_contrasts):
        for kk,dur in enumerate(interested_durations):
            print('ctr:',ctr,' dur:',dur)
            jj = np.squeeze(np.argwhere(potential_durations==interested_durations))
            data_that_condition = {}
            data_that_condition['contrast'] = ctr
            data_that_condition['duration'] = dur
            perf_that_condn = get_performance_for_virtual_session(total_df,N_samplings,N_trials,N_units,ii,jj)
            data_that_condition['performances'] = perf_that_condn
            
            data_all_condns.append(data_that_condition)
    print('saving to ',file_for_pop_size)
    with open(os.path.join(save_loc,file_for_pop_size),'wb') as f:
        pickle.dump(data_all_condns,f)
                
            
            
    