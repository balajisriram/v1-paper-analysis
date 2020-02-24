import pandas as pd
from  datetime import datetime, timedelta
from collections.abc import Iterable
import os

std_data_location =r'C:\Users\bsriram\Desktop\Code\V1PaperAnalysis\Figure1\VarDurCompleted_pd'
rand_state = datetime.date(year=2017,month=11,day=2).toordinal()

def matlab_to_python_datetime(matlab_datenum):
    return datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)

def get_data(ids,data_loc):
    if not isinstance(ids,list):
        ids = [ids]
    output_df = pd.DataFrame()
    for id in ids:
        pickle_file = [f for f in os.listdir(data_loc) if f.startswith(id)]
        current_df = pd.read_pickle(os.path.join(data_loc,pickle_file[0]))
        current_df['subject_id'] = id
        
        # index needs to change on current_df to join it to the output_df
        try:
            max_index = output_df.index[-1]
        except IndexError:
            max_index = 0
            
        current_df.index = current_df.index+max_index
        output_df = pd.concat([ouput_df,current_df])
        
    return output_df
        
def analyze_mouse(id,data_loc=std_data_location,which_days=1,min_trials_per_day=25,analysis_for='varied_contrasts'):
    data= get_data(id,data_loc)
    if analysis_for=='varied_contrasts':
        pass
    elif analysis_for=='varied_contrasts':
        pass
    elif analysis_for=='varied_contrasts':
        pass
    elif analysis_for=='varied_contrasts':
        pass
        
    
if __name__=='__main__':
    df = get_data('218',std_data_location)
    print(df)