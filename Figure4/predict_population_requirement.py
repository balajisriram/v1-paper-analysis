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
import numpy.matlib

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
    
def sample_from_population_simple(df,N_trials,N_units,ctr_idx,dur_idx,):
    remove_list = []
    # make sure that the corresponding df has data
    for idx,row in df.iterrows():
        if row.response_hist[0][ctr_idx][dur_idx].size==0 or row.response_hist[1][ctr_idx][dur_idx].size==0:
            remove_list.append(idx)
    df_removed = df.drop(remove_list)
    if df.count==0:
       print('bad')
       return [],[],[],[],'no_data'
    try:
        sub_df = df_removed.sample(n=N_units,replace=True)
    except:
        pdb.set_trace()
    units_this_sample = sub_df.unit_id
    
    orientations = np.random.choice([0,1],N_trials)
    orientations = np.sort(orientations)
    X = np.full((orientations.size,N_units),np.nan)
    c_vec = np.full((1,N_units),np.nan)
    i_vec = c_vec.copy()
    for jj,unit in enumerate(units_this_sample):
        c_vec[0,jj] = sub_df.iloc[jj].mean_coeff_shortdur
        i_vec[0,jj] = sub_df.iloc[jj].mean_intercept_shortdur
        r_hist = sub_df.iloc[jj].response_hist
        num_trials_0 = np.sum(orientations==0)
        num_trials_1 = np.sum(orientations==1)
        relevant_resp_0 = r_hist[0][ctr_idx][dur_idx]
        relevant_resp_1 = r_hist[1][ctr_idx][dur_idx]
        try:
            X[0:num_trials_0,jj] = sample_from(relevant_resp_0,size=num_trials_0)
            X[num_trials_0:,jj] = sample_from(relevant_resp_1,size=num_trials_1)
        except ValueError:
            pdb.set_trace()
            print('relevant_resp:',relevant_resp)
    C = numpy.matlib.repmat(c_vec,N_trials,1)
    I = numpy.matlib.repmat(i_vec,N_trials,1)
    

    return X,orientations,C,I,'nominal'


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
    
def get_performance_for_virtual_session_simple(total_df,N_samplings,N_trials,N_units,ctr_idx,dur_idx):
    performances_that_condition = []
    performances_that_condition_unweighted = []
    num_spikes_total = []
    num_spikes_differential = []
    for i in tqdm(range(N_samplings)):
        X,y,C,I,reason = sample_from_population_simple(total_df,N_trials,N_units,ctr_idx,dur_idx)
        if reason=='no_data':
            performances_that_condition = []
            return performances_that_condition
        y_pred = np.sum(X*C+I,axis=1)
        y_pred = y_pred>0
        perf = np.sum(y_pred==y)/y.size
        
        C_unweighted = C/np.abs(C)
        y_pred_unweighted = np.sum(X*C_unweighted,axis=1)
        y_pred_unweighted = y_pred_unweighted>0
        perf_unweighted = np.sum(y_pred_unweighted==y)/y.size
        
        num_spikes_total_per_trial = np.squeeze(np.sum(X,axis=1))
        num_spikes_differential_per_trial = np.abs(np.squeeze(np.sum(X*C_unweighted,axis=1)))
        
        performances_that_condition.append(perf)
        performances_that_condition_unweighted.append(perf_unweighted)
        num_spikes_total.append(np.mean(num_spikes_total_per_trial))
        num_spikes_differential.append(np.mean(num_spikes_differential_per_trial))
    return performances_that_condition,performances_that_condition_unweighted,num_spikes_total,num_spikes_differential
    
if __name__=='__main__':
    which = int(sys.argv[1])
    which = which-1
    # total_df = pd.read_pickle('/camhpc/home/bsriram/v1paper/v1-paper-analysis/Figure4/DecodingOfPopulation_onlyConsistent.df')
    # total_df = pd.read_pickle('Figure4\DecodingOfPopulation_onlyConsistent.df')
    total_df = pd.read_pickle('/camhpc/home/bsriram/data/Analysis/DecodingOfPopulation_onlyConsistent.df')
    save_loc = '/camhpc/home/bsriram/data/Analysis/PerfByPopsize'
    # save_loc = r'C:\Users\bsriram\Desktop\Data_V1Paper\Analysis\PopulationDecoding'
    # sample N units and create a session for C = 0.15, dur = 0.1
    potential_n_units = np.arange(1,201,1)
    potential_n_units = np.concatenate((potential_n_units,np.arange(205,401,5)),axis=None)
    potential_n_units = np.concatenate((potential_n_units,np.arange(450,1001,50)),axis=None)
    potential_n_units = np.concatenate((potential_n_units,np.arange(1500,10001,500)),axis=None)
    
    # for N_units in potential_n_units:
    N_units = potential_n_units[which]
    N_trials = 1000
    N_samplings = 10000
    potential_orientations = np.array([-45,45])
    
    potential_contrasts = np.array([0,0.15, 1])
    interested_contrasts = np.array([0.15, 1])
    correct_index_ctrs = [1,2]
    potential_durations = np.array([0.05,0.1,0.15, 0.2])
    interested_durations = np.array([0.05,0.1, 0.15,0.2])
    correct_index_durs = [0,1,2,3]

    print('running for num population==',N_units)
    file_for_pop_size = 'population_{0}.pickle'.format(N_units)
    data_all_condns = []
    for kk,ctr in enumerate(interested_contrasts):
        ii = correct_index_ctrs[kk]
        for ll,dur in enumerate(interested_durations):
            print('ctr:',ctr,' dur:',dur)
            jj = correct_index_durs[ll]
            data_that_condition = {}
            data_that_condition['contrast'] = ctr
            data_that_condition['duration'] = dur
            perf_that_condn,perf_unweighted,n_spikes_tot,n_spikes_diff = get_performance_for_virtual_session_simple(total_df,N_samplings,N_trials,N_units,ii,jj)
            data_that_condition['performances'] = perf_that_condn
            data_that_condition['performances_unweighted'] = perf_unweighted
            data_that_condition['spikes_total'] = n_spikes_tot
            data_that_condition['spikes_differential'] = n_spikes_diff
            
            data_all_condns.append(data_that_condition)
            print('saving to ',file_for_pop_size)
            with open(os.path.join(save_loc,file_for_pop_size),'wb') as f:
                pickle.dump(data_all_condns,f)
                
            
            
    