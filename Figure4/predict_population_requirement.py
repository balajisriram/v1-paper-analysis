import pandas as pd
import pickle
import pdb
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from tqdm import tqdm

sample_from = np.random.choice


total_df = pd.read_pickle('DecodingOfPopulation_onlyConsistent.df')


def sample_from_population(df, N_trials,N_units,ctr_idx,dur_idx,):
    sub_df = df.sample(n=N_units,replace=True)
    units_this_sample = sub_df.unit_id
    
    orientations = np.random.choice([0,1],N_trials)
    X = np.full((orientations.size,N_units),np.nan)

    for ii,ori in enumerate(orientations):
        for jj,unit in enumerate(units_this_sample):
            r_hist = sub_df.iloc[jj].response_hist
            try:
                relevant_resp = r_hist[ori][ctr_idx][dur_idx]
                X[ii,jj] = sample_from(relevant_resp)
            except:
                pdb.set_trace()
    return X,orientations


# sample N units and create a session for C = 0.15, dur = 0.1
potential_n_units = [1,2,3,5,8,10,13,15,18,20,23,25,28,30,32,40,50,64,72,96,108,128,176,256,378,512,756,1024]

N_units = 20
N_trials = 1000
N_samplings = 10
potential_orientations = np.array([-45,45])
potential_contrasts = np.array([0, 0.15, 1])
potential_durations = np.array([0.05,0.1,0.15,0.2])

ctr_idx = 1
dur_idx = 1
performances_that_condition = []
for i in tqdm(range(N_samplings)):
    X,y = sample_from_population(total_df,N_trials,N_units,ctr_idx,dur_idx)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

    X_train = np.insert(X_train,0,1.0,axis=1)
    X_test = np.insert(X_test,0,1.0,axis=1)

    logreg = sm.Logit(y_train,X_train)
    res = logreg.fit(disp=False)
    predicted = res.predict(X_test)
    predicted = (predicted>=0.5)
    perf = np.sum(predicted==y_test)/y_test.size

    performances_that_condition.append(perf)
print(performances_that_condition)